function [Phi, fieldInfo, env] = buildGuidanceField(x, dirSignList, env)
%% ========================================================================
% buildGuidanceField.m
%
% 1) 按论文 Eq. (66) 在“进入 outer activation region”时更新 chi_i；
% 2) 在该 obstacle 的 outer region 内保持 chi_i 不变；
% 3) 退出后允许下一次重新进入时再次选择；
% 4) 调用 computeFieldAtPoint 计算论文 Eq. (70) 的统一导航 GVF。
%% ========================================================================

    p = x(1:2);
    M = env.numObs;
    chi = dirSignList(:);
    if numel(chi) ~= M
        chi = env.chi_init_vec(:);
    end
    chi(chi==0) = env.chi_init_vec(chi==0);

    Fi = inf(M,1);
    n_all = zeros(2,M);
    outerNow = false(M,1);

    for i = 1:M
        obs = env.obstacles(i);
        [Fi(i), gradFi] = smoothPolyObstacleFunction( ...
            p, obs.A_obs, obs.b_obs, env.kappa_vec(i));
        outerNow(i) = Fi(i) < env.d_out_vec(i);

        ng = norm(gradFi);
        if ng > env.grad_eps
            n_all(:,i) = gradFi/ng;
        end
    end

    entryMask = outerNow & ~env.prevOuterMask;
    exitMask  = ~outerNow & env.prevOuterMask;

    rg = norm(env.goal-p);
    if rg > env.goal_dir_eps
        d_g = (env.goal-p)/rg;

        for i = find(entryMask(:))'
            n_i = n_all(:,i);
            if norm(n_i) <= env.grad_eps
                continue;
            end

            % paper Eq. (65): d_o,i = -n_i (local inward normal)
            d_o = -n_i;
            detVal = d_g(1)*d_o(2) - d_g(2)*d_o(1);

            % Paper Eq. (66), together with the stated non-ambiguous
            % |det| condition used to establish Eq. (67).
            if abs(detVal) > env.eps_chi_vec(i)
                chi(i) = sign(detVal);
            end
            % 否则保持此前预置/存储方向不变
        end
    end

    env.prevOuterMask = outerNow;

    seedInfo = struct('dirSignList', chi);
    [Phi, weightSum, fieldMode, details] = computeFieldAtPoint(p, seedInfo, env);

    activeMask = details.activeMask;
    activeIdx = find(activeMask);

    if isempty(activeIdx)
        primaryObsIdx = 0;
    else
        [~,ii] = max(details.wi(activeIdx));
        primaryObsIdx = activeIdx(ii);
    end

    fieldInfo = struct();
    fieldInfo.dirSignList = chi;
    fieldInfo.fieldMode = fieldMode;
    fieldInfo.needAvoid = any(activeMask);
    fieldInfo.activeObsMask = activeMask;
    fieldInfo.activeObsIdxList = activeIdx;
    fieldInfo.numActiveObs = numel(activeIdx);
    fieldInfo.primaryObsIdx = primaryObsIdx;
    fieldInfo.outerObsMask = outerNow;
    fieldInfo.entryMask = entryMask;
    fieldInfo.exitMask = exitMask;
    fieldInfo.weightSum = weightSum;

    fieldInfo.Fi = details.Fi;
    fieldInfo.Vi = details.Vi;
    fieldInfo.wi = details.wi;
    fieldInfo.w0 = details.w0;
    fieldInfo.n_hat_all = details.n_all;
    fieldInfo.t_hat_all = details.t_all;
    fieldInfo.Phi_goal = details.Phi_goal;
    fieldInfo.Phi_obs_all = details.phi_o_all;
end