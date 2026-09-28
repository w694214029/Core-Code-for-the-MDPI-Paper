function [Phi, weightSum, fieldMode, details] = computeFieldAtPoint(p, fieldInfo, env)
%% ========================================================================
% computeFieldAtPoint.m
%
% 在任意位置 p 上计算论文式 (70) 的 blended navigation GVF。
% 该函数是唯一的“点上 GVF 计算器”，机器人当前点、参考轨迹生成与绘图均调用
% 同一函数，避免旧代码中多套场定义不一致的问题。
%
% 对应论文：
%   Eqs. (23)-(25) : F_i, grad F_i
%   Eq. (34)       : V_i = F_i - delta_b
%   Eq. (48)       : goal GVF phi_g
%   Eqs. (51)-(53) : obstacle GVF phi_o,i
%   Eq. (64)       : A(p)
%   Eqs. (68)-(70) : w_i, w_0, Phi
%% ========================================================================

    M = env.numObs;

    if nargin >= 2 && isfield(fieldInfo,'dirSignList') && numel(fieldInfo.dirSignList)==M
        chi = fieldInfo.dirSignList(:);
    else
        chi = env.chi_init_vec(:);
    end
    chi(chi==0) = 1;

    Phi_g = attractiveField(p, env.goal, env.k_g, env.v_g);
    rg = norm(p-env.goal);

    Fi      = inf(M,1);
    Vi      = inf(M,1);
    wi      = zeros(M,1);
    n_all   = zeros(2,M);
    t_all   = zeros(2,M);
    phi_o   = zeros(2,M);
    outerMask = false(M,1);

    % 论文在 p = p_g 时直接定义 Phi(p_g)=0，且不计算 d_g。
    if rg <= env.goal_dir_eps
        Phi = [0;0];
        weightSum = 0;
        fieldMode = 0;
        details = packDetails();
        return;
    end

    d_g = (env.goal-p)/rg;
    Eperp = [0 -1; 1 0];

    for i = 1:M
        obs = env.obstacles(i);
        [Fi(i), gradFi] = smoothPolyObstacleFunction( ...
            p, obs.A_obs, obs.b_obs, env.kappa_vec(i));
        Vi(i) = Fi(i) - env.delta_b;

        % A(p) = {i: F_i(p) < d_out,i}, paper Eq. (64)
        if Fi(i) >= env.d_out_vec(i)
            continue;
        end
        outerMask(i) = true;

        ng = norm(gradFi);
        if ng <= env.grad_eps
            % 理论 Assumption 2 排除了相关带内的梯度奇异点。
            % 若数值上出现异常，则不激活该局部场，避免制造虚假方向。
            continue;
        end

        n_i = gradFi / ng;                    % outward normal
        t_i = chi(i) * (Eperp*n_i);           % Eq. (51)

        c_i = (2/pi) * atan(env.k_o_vec(i) * Vi(i));
        s_i = sqrt(max(1-c_i^2, 0));

        phi_i = env.v_o_vec(i) * (-c_i*n_i + s_i*t_i); % Eq. (53)

        z_dist = (env.d_out_vec(i)-Fi(i)) / ...
                 (env.d_out_vec(i)-env.d_in_vec(i));
        z_dir  = (env.eps_a_vec(i) - d_g'*n_i) / ...
                 (2*env.eps_a_vec(i));

        wi(i) = smoothCutoffCinf(z_dist) * smoothCutoffCinf(z_dir); % Eq. (68)

        n_all(:,i) = n_i;
        t_all(:,i) = t_i;
        phi_o(:,i) = phi_i;
    end

    % Eq. (69)
    w0 = prod(1-wi);

    % Eq. (70): normalized scalar-weighted blend; 不做额外单位向量归一化。
    denom = w0 + sum(wi);
    if denom <= eps
        % 按论文构造理论上 denom 始终严格为正；此处只是数值保护。
        Phi = Phi_g;
    else
        Phi = w0*Phi_g + phi_o*wi;
        Phi = Phi / denom;
    end

    activeMask = wi > env.weight_active_tol;
    weightSum = sum(wi);
    fieldMode = double(any(activeMask));

    details = packDetails();

    function d = packDetails()
        d = struct();
        d.Phi_goal = Phi_g;
        d.Fi = Fi;
        d.Vi = Vi;
        d.wi = wi;
        d.w0 = prod(1-wi);
        d.n_all = n_all;
        d.t_all = t_all;
        d.phi_o_all = phi_o;
        d.outerMask = outerMask;
        d.activeMask = wi > env.weight_active_tol;
        d.dirSignList = chi;
    end
end