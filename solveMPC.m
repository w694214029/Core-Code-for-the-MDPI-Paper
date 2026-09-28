function [vCmd,omegaCmd,mpcInfo] = solveMPC(x,fieldInfo,env)
%% ========================================================================
% Paper-aligned MPC--DCBF controller for the supplied September 28 paper:
%   Eq. (6)        sampled-data forward-Euler prediction model;
%   Eqs. (74)-(76) admissibility-checked GVF reference;
%   Eq. (77)       tracking and increment cost;
%   Eq. (79)       hard DCBF inequality;
%   Eq. (81)       constrained finite-horizon MPC problem.
%
% If no feasible optimal solution is obtained, retain the previously applied
% input, exactly as specified in Section 4 of the supplied paper.
%% ========================================================================

    N = env.N;
    Nc = env.Nc;
    uPrev = [env.v_now;env.omega_now];

    % Paper Eqs. (74)--(76).
    ref = generateReferenceFromField(x,fieldInfo,env);

    % Paper Eqs. (81d)--(81e).
    lb = repmat([env.v_min;env.omega_min],Nc,1);
    ub = repmat([env.v_max;env.omega_max],Nc,1);
    [Aineq,bineq] = buildIncrementConstraints(uPrev,env);

    U0 = buildInitialGuess(ref,uPrev,env);
    options = optimoptions('fmincon', ...
        'Algorithm',env.solver_algorithm, ...
        'Display',env.solver_display, ...
        'MaxIterations',env.solver_max_iterations, ...
        'MaxFunctionEvaluations',env.solver_max_fun_evals, ...
        'OptimalityTolerance',env.solver_opt_tol, ...
        'ConstraintTolerance',env.solver_con_tol, ...
        'StepTolerance',env.solver_step_tol);

    objective = @(U) mpcObjective(U,x,ref,env);
    nonlinearConstraints = @(U) mpcNonlinearConstraints(U,x,env);
    [U,exitflag,output,J] = runFmincon( ...
        U0,objective,nonlinearConstraints,Aineq,bineq,lb,ub,options);

    if ~decisionIsFeasible(U,exitflag,nonlinearConstraints,env)
        % Paper Section 4: if no feasible optimal solution is obtained in
        % the current update period, retain the previously applied input.
        vCmd = uPrev(1);
        omegaCmd = uPrev(2);
        Uhold = repmat(uPrev,Nc,1);
        [Xpred,Ufull] = predictTrajectory(Uhold,x,env);
        [minResidual,residuals] = computeDCBFResiduals(Xpred,env);
        hPred = evaluateBarrierTrajectory(Xpred,env);

        mpcInfo = struct();
        mpcInfo.success = false;
        mpcInfo.usedPreviousInput = true;
        mpcInfo.exitflag = exitflag;
        mpcInfo.output = output;
        mpcInfo.objective = J;
        mpcInfo.ref = ref;
        mpcInfo.U_opt = U;
        mpcInfo.U_full = Ufull;
        mpcInfo.X_pred = Xpred;
        mpcInfo.h_pred = hPred;
        mpcInfo.dcbf_residuals = residuals;
        mpcInfo.min_dcbf_residual = minResidual;
        mpcInfo.heading_error_to_ref = wrapToPiLocal(ref.theta(2)-x(3));
        return;
    end

    % Paper Section 4: only the first feasible optimal input is applied.
    vCmd = U(1);
    omegaCmd = U(2);

    [Xpred,Ufull] = predictTrajectory(U,x,env);
    [minResidual,residuals] = computeDCBFResiduals(Xpred,env);
    hPred = evaluateBarrierTrajectory(Xpred,env);

    mpcInfo = struct();
    mpcInfo.success = true;
    mpcInfo.usedPreviousInput = false;
    mpcInfo.exitflag = exitflag;
    mpcInfo.output = output;
    mpcInfo.objective = J;
    mpcInfo.ref = ref;
    mpcInfo.U_opt = U;
    mpcInfo.U_full = Ufull;
    mpcInfo.X_pred = Xpred;
    mpcInfo.h_pred = hPred;
    mpcInfo.dcbf_residuals = residuals;
    mpcInfo.min_dcbf_residual = minResidual;
    mpcInfo.heading_error_to_ref = wrapToPiLocal(ref.theta(2)-x(3));
end

%% ========================================================================
% Paper Eq. (77)
%% ========================================================================
function J = mpcObjective(U,x0,ref,env)
    [X,~] = predictTrajectory(U,x0,env);
    J = 0;

    for j = 0:env.N-1
        xRef = [ref.x(j+1);ref.y(j+1);ref.theta(j+1)];
        errorState = X(:,j+1)-xRef;
        errorState(3) = wrapToPiLocal(errorState(3));
        J = J + errorState'*env.Q*errorState;
    end

    xRefN = [ref.x(env.N+1);ref.y(env.N+1);ref.theta(env.N+1)];
    errorN = X(:,env.N+1)-xRefN;
    errorN(3) = wrapToPiLocal(errorN(3));
    J = J + errorN'*env.Qf*errorN;

    uPrev = [env.v_now;env.omega_now];
    for j = 1:env.Nc
        u = U((j-1)*2+(1:2));
        if j == 1
            du = u-uPrev;
        else
            uPrevious = U((j-2)*2+(1:2));
            du = u-uPrevious;
        end
        J = J + du'*env.R*du;
    end
end

%% ========================================================================
% Paper Eqs. (79) and (81f)
%% ========================================================================
function [c,ceq] = mpcNonlinearConstraints(U,x0,env)
    [X,~] = predictTrajectory(U,x0,env);
    c = [];

    for j = 0:env.Nb-1
        h0 = evaluateBarrierFunctions(X(1:2,j+1),env);
        h1 = evaluateBarrierFunctions(X(1:2,j+2),env);
        c = [c;(1-env.gamma_vec).*h0 + env.rho_vec - h1]; %#ok<AGROW>
    end

    if env.enforce_workspace_bounds
        for j = 1:env.N
            px = X(1,j+1);
            py = X(2,j+1);
            c = [c;px-env.xMax;env.xMin-px;py-env.yMax;env.yMin-py]; %#ok<AGROW>
        end
    end
    ceq = [];
end

%% ========================================================================
% Paper Eq. (6), with held inputs after the control horizon.
%% ========================================================================
function [X,Ufull] = predictTrajectory(U,x0,env)
    X = zeros(3,env.N+1);
    Ufull = zeros(2,env.N);
    X(:,1) = x0;

    for j = 1:env.N
        index = min(j,env.Nc);
        u = U((index-1)*2+(1:2));
        Ufull(:,j) = u;
        theta = X(3,j);
        X(1,j+1) = X(1,j)+env.dt*u(1)*cos(theta);
        X(2,j+1) = X(2,j)+env.dt*u(1)*sin(theta);
        X(3,j+1) = wrapToPiLocal(X(3,j)+env.dt*u(2));
    end
end

%% ========================================================================
function [Aineq,bineq] = buildIncrementConstraints(uPrev,env)
    nU = 2*env.Nc;
    D = zeros(nU,nU);
    d0 = zeros(nU,1);

    for j = 1:env.Nc
        rows = (j-1)*2+(1:2);
        D(rows,rows) = eye(2);
        if j == 1
            d0(rows) = uPrev;
        else
            previousRows = (j-2)*2+(1:2);
            D(rows,previousRows) = -eye(2);
        end
    end

    deltaMax = repmat([env.dv_max;env.domega_max],env.Nc,1);
    deltaMin = -deltaMax;
    Aineq = [D;-D];
    bineq = [deltaMax+d0;-deltaMin-d0];
end

%% ========================================================================
% This is only a numerical initial point for fmincon; it does not alter
% the paper objective or constraints.
%% ========================================================================
function U0 = buildInitialGuess(ref,uPrev,env)
    U0 = zeros(2*env.Nc,1);
    last = uPrev;

    for j = 1:env.Nc
        desired = [ref.v(min(j,numel(ref.v)));ref.omega(min(j,numel(ref.omega)))];
        u = applyIncrementAndInputLimits(desired,last,env);
        U0((j-1)*2+(1:2)) = u;
        last = u;
    end
end

function u = applyIncrementAndInputLimits(desired,last,env)
    dv = min(max(desired(1)-last(1),-env.dv_max),env.dv_max);
    domega = min(max(desired(2)-last(2),-env.domega_max),env.domega_max);
    u = last+[dv;domega];
    u(1) = min(max(u(1),env.v_min),env.v_max);
    u(2) = min(max(u(2),env.omega_min),env.omega_max);
end

%% ========================================================================
function [minResidual,residuals] = computeDCBFResiduals(X,env)
    residuals = zeros(env.numObs,env.Nb);
    for j = 0:env.Nb-1
        h0 = evaluateBarrierFunctions(X(1:2,j+1),env);
        h1 = evaluateBarrierFunctions(X(1:2,j+2),env);
        residuals(:,j+1) = h1-(1-env.gamma_vec).*h0-env.rho_vec;
    end
    minResidual = min(residuals(:));
end

function hPred = evaluateBarrierTrajectory(X,env)
    hPred = zeros(env.numObs,env.N+1);
    for j = 1:env.N+1
        hPred(:,j) = evaluateBarrierFunctions(X(1:2,j),env);
    end
end

%% ========================================================================
function [U,exitflag,output,J] = runFmincon(U0,objective,nonlinearConstraints,Aineq,bineq,lb,ub,options)
    U = [];
    exitflag = -999;
    output = struct();
    J = inf;
    try
        [U,J,exitflag,output] = fmincon( ...
            objective,U0,Aineq,bineq,[],[],lb,ub,nonlinearConstraints,options);
    catch ME
        output.message = ME.message;
    end
end

%% ========================================================================
function isFeasible = decisionIsFeasible(U,exitflag,nonlinearConstraints,env)
    isFeasible = ~isempty(U) && exitflag > 0;
    if ~isFeasible
        return;
    end
    try
        [c,ceq] = nonlinearConstraints(U);
        isFeasible = all(c <= env.dcbf_feas_tol) && ...
            all(abs(ceq) <= env.dcbf_feas_tol);
    catch
        isFeasible = false;
    end
end

function angle = wrapToPiLocal(angle)
    angle = mod(angle+pi,2*pi)-pi;
end