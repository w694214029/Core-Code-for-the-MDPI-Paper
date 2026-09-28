function [Fval, gradF] = smoothPolyObstacleFunction(p, A, b_rhs, kappa)
%% ========================================================================
% smoothPolyObstacleFunction.m
%
% LSE obstacle model corresponding to paper Eqs. (23)-(25).
%
% Paper notation:
%   Each polygon half-space is written as
%       a_l^T p + b_l^{paper} <= 0,
%   and the associated signed half-space function is
%       ell_l(p) = a_l^T p + b_l^{paper}.
%
% Implementation convention:
%   This MATLAB function uses the equivalent standard H-representation
%       A*p <= b_rhs,
%   so that
%       psi_l(p) = a_l^T p - b_rhs,l,
%   with the parameter correspondence
%       b_rhs,l = -b_l^{paper}.
%
% Therefore, psi_l(p) in the code is exactly the same half-space function
% as ell_l(p) in the paper, expressed under an equivalent sign convention.
%
% The implemented LSE function and its gradient are
%
%   F(p) = (1/kappa) * log( (1/N) * sum_l exp(kappa*psi_l(p)) ),
%
%   grad F(p) = sum_l lambda_l a_l,
%
% where
%
%   lambda_l =
%       exp(kappa*psi_l(p)) /
%       sum_q exp(kappa*psi_q(p)).
%
% A numerically stable log-sum-exp evaluation is used below to avoid
% exponential overflow. The mathematical definition is unchanged.
%% ========================================================================

    psi = A*p - b_rhs;

    z = kappa * psi;
    zmax = max(z);
    ez = exp(z - zmax);
    sum_ez = sum(ez);

    Fval = (zmax + log(sum_ez / numel(z))) / kappa;

    lambda = ez / sum_ez;
    gradF = A' * lambda;
end