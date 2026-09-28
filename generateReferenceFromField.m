function ref = generateReferenceFromField(x,fieldInfo,env)
%% ========================================================================
% Reference generation in the supplied APF_Obstacle_Avoidance_Sensors(1)
% manuscript, Eqs. (74)--(76).
%
% At each step, accept the forward-Euler GVF candidate only when it lies in
% D_b (and hence W). Once a candidate is rejected, hold the last admissible
% reference point and set every remaining reference velocity to zero.
%% ========================================================================

    N = env.N;
    Ts = env.dt;

    ref.x = zeros(1,N+1);
    ref.y = zeros(1,N+1);
    ref.theta = zeros(1,N+1);
    ref.v = zeros(1,N);
    ref.omega = zeros(1,N);  % numerical initial-guess data only
    ref.stepAccepted = zeros(1,N);  % alpha_{j|k} in Eq. (75)
    ref.min_h = inf(1,N+1);
    ref.stopped = false;
    ref.stopStep = NaN;

    pRef = x(1:2);
    thetaRef = x(3);
    ref.x(1) = pRef(1);
    ref.y(1) = pRef(2);
    ref.theta(1) = thetaRef;

    [isAdmissible,ref.min_h(1)] = isInDesignSafeDomain(pRef,env);
    if ~isAdmissible || ~isInWorkspace(pRef,env)
        error('Paper Eq. (74) requires the measured position p_k to belong to D_b.');
    end

    % Section 4: update the circulation state at the measured position,
    % then hold it fixed through this reference/MPC horizon.
    fixedFieldInfo = struct('dirSignList',fieldInfo.dirSignList);

    for j = 1:N
        [Phi,~,~,~] = computeFieldAtPoint(pRef,fixedFieldInfo,env);
        pCandidate = pRef + Ts*Phi;
        [candidateInDb,minH] = isInDesignSafeDomain(pCandidate,env);

        if ~(candidateInDb && isInWorkspace(pCandidate,env))
            % Paper Section 4: preserve the last admissible reference
            % position and set alpha and reference velocity to zero.
            ref.x(j+1:end) = pRef(1);
            ref.y(j+1:end) = pRef(2);
            ref.theta(j+1:end) = thetaRef;
            ref.v(j:end) = 0;
            ref.omega(j:end) = 0;
            ref.stepAccepted(j:end) = 0;
            ref.min_h(j+1:end) = ref.min_h(j);
            ref.stopped = true;
            ref.stopStep = j;
            return;
        end

        % The accepted candidate has alpha_{j|k}=1, so Eq. (75) gives
        % nu_ref = Phi(p_ref). Heading aligns with the nonzero velocity.
        speedRef = norm(Phi);
        if speedRef > env.ref_field_eps
            thetaNext = atan2(Phi(2),Phi(1));
        else
            thetaNext = thetaRef;
        end

        ref.x(j+1) = pCandidate(1);
        ref.y(j+1) = pCandidate(2);
        ref.theta(j+1) = thetaNext;
        ref.v(j) = speedRef;
        ref.omega(j) = wrapToPiLocal(thetaNext-thetaRef)/Ts;
        ref.stepAccepted(j) = 1;
        ref.min_h(j+1) = minH;

        pRef = pCandidate;
        thetaRef = thetaNext;
    end
end

function tf = isInWorkspace(p,env)
% The map-redacted caller supplies only the generic workspace interface.
    required = {'xMin','xMax','yMin','yMax'};
    if ~all(isfield(env,required))
        error('Workspace bounds are required to enforce Paper Eq. (74).');
    end
    tf = p(1) >= env.xMin && p(1) <= env.xMax && ...
         p(2) >= env.yMin && p(2) <= env.yMax;
end

function angle = wrapToPiLocal(angle)
    angle = mod(angle+pi,2*pi)-pi;
end