function [xNext,dirSignList,env,sample] = main(x,dirSignList,env)
%% ========================================================================
% main.m -- map-interface-redacted one-sample GVF + MPC-DCBF dispatcher
%
% Paper-aligned sequence:
%   Eq. (66), Eqs. (68)--(70): update the navigation GVF;
%   Eqs. (74)--(76): form the admissibility-checked reference;
%   Eqs. (77), (79), (81): solve MPC--DCBF or retain the prior input;
%   Eq. (6): apply one forward-Euler sampled-data state update.
%% ========================================================================

    % --------------------------------------------------------------------
    % MAP INTERFACE -- deliberately retained as comments for review only.
    % No map file is opened or referenced by active code in this package.
    %
    % mapFile = '<path-to-map-file>';
    % mapData = load(mapFile);
    % env = setupEnvironmentFromMap(mapData,cfg);
    % x = env.x0;
    % dirSignList = env.chi_init_vec;
    % --------------------------------------------------------------------

    % --------------------------------------------------------------------
    % MAP-DEPENDENT rendering and output are also intentionally commented:
    %
    % drawScene(...);
    % writeVideo(...);
    % saveFigureTriple(...);
    % saveSimulationResults(...);
    % --------------------------------------------------------------------

    % Eqs. (66), (68)--(70): update circulation directions and GVF.
    [Phi,fieldInfo,env] = buildGuidanceField(x,dirSignList,env);
    dirSignList = fieldInfo.dirSignList;

    % Eqs. (74)--(81): solve MPC--DCBF or retain the previous input.
    [vCmd,omegaCmd,mpcInfo] = solveMPC(x,fieldInfo,env);

    % Paper Eq. (6): zero-order-hold sampled-data plant update.
    xNext = x;
    xNext(1) = x(1) + env.dt*vCmd*cos(x(3));
    xNext(2) = x(2) + env.dt*vCmd*sin(x(3));
    xNext(3) = wrapToPiLocal(x(3) + env.dt*omegaCmd);
    env.v_now = vCmd;
    env.omega_now = omegaCmd;

    % Generic diagnostic only; its helper and concrete environment are absent.
    [hNow,~] = evaluateBarrierFunctions(xNext(1:2),env);
    sample = struct( ...
        'Phi',Phi, ...
        'fieldInfo',fieldInfo, ...
        'v_cmd',vCmd, ...
        'omega_cmd',omegaCmd, ...
        'mpc_success',mpcInfo.success, ...
        'used_previous_input',mpcInfo.usedPreviousInput, ...
        'min_h',min(hNow), ...
        'min_dcbf_residual',mpcInfo.min_dcbf_residual, ...
        'reference',mpcInfo.ref);
end

function angle = wrapToPiLocal(angle)
    angle = mod(angle+pi,2*pi)-pi;
end