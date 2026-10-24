% MATLAB smoke test for the ocp++ MEX wrapper (double_integrator config).
% Exercises the full command table from docs/plans/MATLAB_MEX_PLAN.md
% section 4 on the double integrator.
%
% Usage (from the repo root):
%   matlab -batch "addpath('build'); test_mex"
%
% Requires the ocp_mex_double_integrator MEX (cmake -DOCP_BUILD_MEX=ON,
% see mex/README.md).

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'build'));

N  = 10;
Ts = 0.1;
q0 = 1.0;  v0 = 0.5;   % matches DoubleIntegrator::initial_state()

% ---- info ------------------------------------------------------------
i = ocp_mex_double_integrator('info');
assert(i.nx == 2 && i.nu == 1, 'nx/nu mismatch');
assert(i.ng == 1 && i.ne == 0 && i.nl == 1, 'stage constraint dims');
assert(i.ng_t == 0 && i.ne_t == 1 && i.nl_t == 0, 'terminal constraint dims');
assert(i.nbx == 2 && i.nbu == 1 && i.nbx_t == 2, 'box counts');
assert(i.fixed_initial_state == 1, 'fixed initial state');
assert(strcmp(i.solver, 'double_integrator'), 'solver name');

% ---- init ------------------------------------------------------------
assert(ocp_mex_double_integrator('init', N), 'init must succeed');

% ---- options: discovery + get/set ------------------------------------
opts = ocp_mex_double_integrator('options');
assert(iscell(opts) && ~isempty(opts), 'options must be a cell of names');
assert(any(strcmp(opts, 'max_iter')), 'max_iter not discoverable');
assert(any(strcmp(opts, 'glob.alpha_min')), 'dotted glob.* names missing');
assert(ocp_mex_double_integrator('opt_get', 'max_iter') == 20, 'default max_iter');
assert(ocp_mex_double_integrator('opt_set', 'max_iter', 40), 'opt_set max_iter');
assert(ocp_mex_double_integrator('opt_get', 'max_iter') == 40, 'opt_set not live?');
assert(ocp_mex_double_integrator('opt_set', 'glob.alpha_min', 0.1), 'opt_set glob.alpha_min');
assert(abs(ocp_mex_double_integrator('opt_get', 'glob.alpha_min') - 0.1) < 1e-15, ...
       'dotted set not live');
assert(ocp_mex_double_integrator('opt_set', 'max_iter', 20), 'restore max_iter');

% ---- warm start (forward simulation of the constant-acceleration input
%      that satisfies the terminal equality q_N + v_N = 1 exactly) ------
den = Ts^2 * N * (N - 1) / 2 + Ts * N;
a   = (1 - (q0 + v0) - Ts * N * v0) / den;
u   = a * ones(1, N);
xq  = zeros(1, N + 1);  xv = zeros(1, N + 1);
xq(1) = q0;  xv(1) = v0;
for k = 1:N
    xq(k + 1) = xq(k) + Ts * xv(k);
    xv(k + 1) = xv(k) + Ts * a;
end
assert(ocp_mex_double_integrator('set', 'u', u), 'set u rejected');
assert(ocp_mex_double_integrator('set', 'x', [xq; xv]), 'set x rejected');
% rejected writes must raise a MEX error
try
    ocp_mex_double_integrator('set', 'x', zeros(2, N));  % wrong extent
    error('wrong-extent set must fail');
catch
end
try
    ocp_mex_double_integrator('set', 'lambda_dyn', zeros(2, N));  % read-only
    error('read-only field set must fail');
catch
end

% ---- solve ------------------------------------------------------------
[status, cost] = ocp_mex_double_integrator('solve');
assert(status == 1, 'expected kSolved (status == 1), got %d', status);
assert(isfinite(cost), 'cost must be finite');

% ---- read back --------------------------------------------------------
x = ocp_mex_double_integrator('get', 'x');
assert(size(x, 1) == 2 && size(x, 2) == N + 1, 'x extents');
assert(abs(x(1,1) - q0) < 1e-12 && abs(x(2,1) - v0) < 1e-12, 'x_0 must stay fixed');
assert(abs(x(1, end) + x(2, end) - 1) < 1e-8, 'terminal equality q_N + v_N = 1');

sol = ocp_mex_double_integrator('solution');
assert(isequal(sol.x, x), 'solution.x must match get x');
assert(numel(sol.lambda_eq_term) == 1, 'lambda_eq_term extent');
assert(abs(sol.cost_value - cost) < 1e-12, 'solution cost mismatch');
assert(sol.status == status, 'solution status mismatch');
assert(sol.N == N, 'solution N mismatch');

% zero-extent fields come back empty (e.g. lambda_eq_stage: ne == 0)
assert(numel(ocp_mex_double_integrator('get', 'lambda_eq_stage')) == 0, ...
       'zero-extent field must be empty');

% ---- stats -------------------------------------------------------------
st = ocp_mex_double_integrator('stats');
assert(st.iter >= 1, 'at least one iteration recorded');
assert(size(st.rows, 2) == 10, 'stats rows must have 10 columns');
assert(st.status == status, 'stats status mismatch');

% opt_dump must print without error
ocp_mex_double_integrator('opt_dump');

% unknown commands must raise a MEX error
try
    ocp_mex_double_integrator('bogus_command');
    error('unknown command must fail');
catch
end

disp('All MEX smoke checks passed.');
