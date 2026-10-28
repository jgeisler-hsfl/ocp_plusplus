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
assert(any(strcmp(opts, 'options.max_iter')), 'options.max_iter not discoverable');
assert(any(strcmp(opts, 'options.glob.alpha_min')), 'dotted glob.* names missing');
assert(any(strcmp(opts, 'qp.options.warm_start')), 'qp.options.* names missing');
assert(any(strcmp(opts, 'scaling.options.scale_objective')), ...
       'scaling.options.* names missing');
assert(ocp_mex_double_integrator('opt_get', 'options.max_iter') == 20, 'default max_iter');
assert(ocp_mex_double_integrator('opt_set', 'options.max_iter', 40), 'opt_set max_iter');
assert(ocp_mex_double_integrator('opt_get', 'options.max_iter') == 40, 'opt_set not live?');
assert(ocp_mex_double_integrator('opt_set', 'options.glob.alpha_min', 0.1), ...
       'opt_set glob.alpha_min');
assert(abs(ocp_mex_double_integrator('opt_get', 'options.glob.alpha_min') - 0.1) < 1e-15, ...
       'dotted set not live');
assert(ocp_mex_double_integrator('opt_set', 'options.max_iter', 20), 'restore max_iter');

% component option surfaces (qp.options.*, scaling.options.*): set/get
% round-trip on the live solver
assert(ocp_mex_double_integrator('opt_set', 'qp.options.warm_start', 3), ...
       'opt_set qp.options.warm_start');
assert(ocp_mex_double_integrator('opt_get', 'qp.options.warm_start') == 3, ...
       'qp.options.warm_start round-trip');
assert(ocp_mex_double_integrator('opt_set', 'qp.options.warm_start', 0), ...
       'restore qp.options.warm_start');
assert(ocp_mex_double_integrator('opt_set', 'scaling.options.scale_objective', 1), ...
       'opt_set scaling.options.scale_objective');
assert(ocp_mex_double_integrator('opt_get', 'scaling.options.scale_objective') == 1, ...
       'scaling.options round-trip (bool)');
assert(ocp_mex_double_integrator('opt_set', 'scaling.options.scale_objective', 0), ...
       'restore scaling.options.scale_objective');

% struct opt_set: field nesting mirrors the dotted names; unknown fields
% raise a warning and are skipped (partial structs are fine)
lastwarn('clear');
opt.options.max_iter = 40;
opt.options.glob.alpha_min = 0.1;
opt.qp.options.warm_start = 2;
opt.bogus = 5;
ocp_mex_double_integrator('opt_set', opt);
assert(ocp_mex_double_integrator('opt_get', 'options.max_iter') == 40, ...
       'struct opt_set not live');
assert(abs(ocp_mex_double_integrator('opt_get', 'options.glob.alpha_min') - 0.1) < 1e-15, ...
       'struct opt_set glob not live');
assert(ocp_mex_double_integrator('opt_get', 'qp.options.warm_start') == 2, ...
       'struct opt_set qp not live');
[~, wid] = lastwarn;
assert(strcmp(wid, 'ocp_mex:UnknownOption'), 'unknown struct field must warn');
assert(ocp_mex_double_integrator('opt_set', 'options.max_iter', 20), 'restore max_iter');
assert(ocp_mex_double_integrator('opt_set', 'options.glob.alpha_min', 0.05), ...
       'restore glob.alpha_min');
assert(ocp_mex_double_integrator('opt_set', 'qp.options.warm_start', 0), ...
       'restore qp warm_start');

% out-of-range opt_set: distinct error id (not UnknownOption), value
% not written
try
    ocp_mex_double_integrator('opt_set', 'qp.options.warm_start', 7);
    error('out-of-range opt_set must fail');
catch ME
    assert(strcmp(ME.identifier, 'ocp_mex:OptionOutOfRange'), ...
           'expected OptionOutOfRange error id');
end
assert(ocp_mex_double_integrator('opt_get', 'qp.options.warm_start') == 0, ...
       'rejected write must leave the option untouched');

% out-of-range struct field: warning (skipped), symmetric with unknown
lastwarn('clear');
opt_bad.qp.options.warm_start = 9;
ocp_mex_double_integrator('opt_set', opt_bad);
[~, wid] = lastwarn;
assert(strcmp(wid, 'ocp_mex:OptionOutOfRange'), ...
       'out-of-range struct field must warn');
assert(ocp_mex_double_integrator('opt_get', 'qp.options.warm_start') == 0, ...
       'out-of-range struct field must be skipped');

% ---- parameters (duck-typed params() hook on DoubleIntegrator) --------
pn = ocp_mex_double_integrator('param_names');
assert(iscell(pn) && numel(pn) == 4, 'param_names must list 4 leaves');
assert(any(strcmp(pn, 'Ts')) && any(strcmp(pn, 'w_stage')), ...
       'param names incomplete');

e_ts = ocp_mex_double_integrator('param_extent', 'Ts');
assert(isequal(e_ts, [1 1]), 'Ts extent must be 1 x 1');
e_w = ocp_mex_double_integrator('param_extent', 'w_stage');
assert(isequal(e_w, [3 1]), 'w_stage extent must be 3 x 1');

ts = ocp_mex_double_integrator('param_get', 'Ts');
assert(abs(ts - 0.1) < 1e-15, 'Ts default mismatch');
w = ocp_mex_double_integrator('param_get', 'w_stage');
assert(isequal(w, [1.0; 0.1; 0.01]), 'w_stage default mismatch');

assert(ocp_mex_double_integrator('param_set', 'Ts', 0.2), 'param_set Ts');
assert(abs(ocp_mex_double_integrator('param_get', 'Ts') - 0.2) < 1e-15, ...
       'param_set Ts not live');
assert(ocp_mex_double_integrator('param_set', 'Ts', 0.1), 'restore Ts');

% matrix leaf round-trip
assert(ocp_mex_double_integrator('param_set', 'w_stage', [4.0; 0.5; 0.25]), ...
       'param_set w_stage');
w2 = ocp_mex_double_integrator('param_get', 'w_stage');
assert(isequal(w2, [4.0; 0.5; 0.25]), 'w_stage round-trip');
assert(ocp_mex_double_integrator('param_set', 'w_stage', [1.0; 0.1; 0.01]), ...
       'restore w_stage');

% wrong extent must raise a MEX error
try
    ocp_mex_double_integrator('param_set', 'w_stage', [1; 2]);
    error('wrong-extent param_set must fail');
catch
end
% unknown name must raise a MEX error
try
    ocp_mex_double_integrator('param_get', 'nope');
    error('unknown param must fail');
catch
end
% out-of-range param_set (Ts is bounded to stay strictly positive):
% distinct error id, leaf untouched
try
    ocp_mex_double_integrator('param_set', 'Ts', 0.0);
    error('out-of-range param_set must fail');
catch ME
    assert(strcmp(ME.identifier, 'ocp_mex:ParamOutOfRange'), ...
           'expected ParamOutOfRange error id');
end
assert(abs(ocp_mex_double_integrator('param_get', 'Ts') - 0.1) < 1e-15, ...
       'rejected param_set must leave the leaf untouched');

% param_dump must print without error
ocp_mex_double_integrator('param_dump');

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
cost2 = ocp_mex_double_integrator('solve');   % single-output form
assert(abs(cost2 - cost) < 1e-12, 'single-output solve cost mismatch');

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
[st, ststatus] = ocp_mex_double_integrator('stats');
assert(ststatus == status, 'stats status mismatch');
assert(isstruct(st) && numel(st) >= 1, 'stats must be a non-empty struct array');
assert(isfield(st, 'res_stat') && isfield(st, 'funnel_width'), ...
       'stats must carry one field per column name');
assert(numel(st) >= 1, 'at least one iteration recorded');
alpha = ocp_mex_double_integrator('stat_col', 'alpha');
assert(size(alpha, 1) == numel(st) && size(alpha, 2) == 1, 'stat_col extent');
assert(isequal(st.alpha, alpha), 'struct-array column must match stat_col');
assert(all(alpha >= 0 & alpha <= 1), 'alpha in [0, 1]');
st2 = ocp_mex_double_integrator('stats');   % single-output form
assert(isequal(st2.res_stat, st.res_stat), 'single-output stats mismatch');
try
    ocp_mex_double_integrator('stat_col', 'bogus');
    error('unknown stat_col must fail');
catch
end

% opt_dump must print without error
ocp_mex_double_integrator('opt_dump');

% unknown commands must raise a MEX error
try
    ocp_mex_double_integrator('bogus_command');
    error('unknown command must fail');
catch
end

% optional outputs: bare calls without assignment must also work
ocp_mex_double_integrator('init', N);
ocp_mex_double_integrator('set', 'u', u);

disp('All MEX smoke checks passed.');
