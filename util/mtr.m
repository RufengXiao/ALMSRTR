function [x, cost, info, options] = mtr(problem, x, options)

M = problem.M;
H = problem.H;
% R = problem.R;

% Define some strings for display
tcg_stop_reason = {'negative curvature',...
                   'exceeded trust region',...
                   'reached target residual-kappa (linear)',...
                   'reached target residual-theta (superlinear)',...
                   'maximum inner iterations',...
                   'model increased'};

% Set local defaults here
localdefaults.verbosity = 2;
localdefaults.maxtime = inf;
localdefaults.miniter = 3;
localdefaults.maxiter = 1000;
localdefaults.mininner = 1;
localdefaults.maxinner = M.dim();
localdefaults.tolgradnorm = 1e-6;
localdefaults.kappa = 0.1;
localdefaults.theta = 1.0;
localdefaults.rho_prime = 0.1;
localdefaults.useRand = false;
localdefaults.rho_regularization = 1e3;

% Merge global and local defaults, then merge w/ user options, if any.
localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
if ~exist('options', 'var') || isempty(options)
    options = struct();
end
options = mergeOptions(localdefaults, options);

% Set default Delta_bar and Delta0 separately to deal with additional
% logic: if Delta_bar is provided but not Delta0, let Delta0 automatically
% be some fraction of the provided Delta_bar.
if ~isfield(options, 'Delta_bar')
    if isfield(M, 'typicaldist')
        options.Delta_bar = M.typicaldist();
    else
        options.Delta_bar = sqrt(M.dim());
    end 
end
if ~isfield(options,'Delta0')
    options.Delta0 = options.Delta_bar / 8;
end

% It is sometimes useful to check what the actual option values are.
if options.verbosity >= 3
    disp(options);
end

ticstart = tic();

% Create a store database and get a key for the current x
storedb = StoreDB(options.storedepth);
key = storedb.getNewKey();

%% Initializations

% k counts the outer (TR) iterations. The semantic is that k counts the
% number of iterations fully executed so far.
k = 0;

% Initialize solution and companion measures: f(x), fgrad(x)
[fx, fgradx] = getCostGrad(problem, x, storedb, key);
norm_grad = M.norm(x, fgradx);

%% Direct stop
if norm_grad < options.tolgradnorm
    cost = fx;
    info.iter = 0;
    info.Delta = options.Delta0;
    info.gradnorm = norm_grad;
    return
end

% Initialize Hessian
Hx_data = H(x);
Hx = @(eta) st_proj(x,Hx_data{1} * eta + Hx_data{2} .* eta + eta * Hx_data{3});

% Initialize trust-region radius
Delta = options.Delta0;

% Save stats in a struct array info, and preallocate.
if ~exist('used_cauchy', 'var')
    used_cauchy = [];
end
stats = savestats(problem, x, storedb, key, options, k, fx, norm_grad, Delta, ticstart);
info(1) = stats;
info(min(10000, options.maxiter+1)).iter = [];

% ** Display:
if options.verbosity == 2
   fprintf(['%3s %3s      %5s                %5s     ',...
            'f: %+e   |grad|: %e\n'],...
           '   ','   ','     ','     ', fx, norm_grad);
elseif options.verbosity > 2
   fprintf('************************************************************************\n');
   fprintf('%3s %3s    k: %5s     num_inner: %5s     %s\n',...
           '','','______','______','');
   fprintf('       f(x) : %+e       |grad| : %e\n', fx, norm_grad);
   fprintf('      Delta : %f\n', Delta);
end

% To keep track of consecutive radius changes, so that we can warn the
% user if it appears necessary.
consecutive_TRplus = 0;
consecutive_TRminus = 0;


% **********************
% ** Start of TR loop **
% **********************
while true
    
    % Start clock for this outer iteration
    ticstart = tic();

    % Apply the hook function if there is one: this allows external code to
    % move x to another point. If the point is changed (indicated by a true
    % value for the boolean 'hooked'), we update our knowledge about x.
    [x, key, info, hooked] = applyHook(problem, x, storedb, key, options, info, k+1);
    if hooked
        [fx, fgradx] = getCostGrad(problem, x, storedb, key);
        norm_grad = M.norm(x, fgradx);
    end
    
    % Run standard stopping criterion checks
    [stop, reason] = stoppingcriterion(problem, x, options, info, k+1);
    
    % If the stopping criterion that triggered is the tolerance on the
    % gradient norm but we are using randomization, make sure we make at
    % least miniter iterations to give randomization a chance at escaping
    % saddle points.
    if stop == 2 && options.useRand && k < options.miniter
        stop = 0;
    end
    
    if stop
        if options.verbosity >= 1
            fprintf([reason '\n']);
        end
        break;
    end

    if options.verbosity > 2 || options.debug > 0
        fprintf('************************************************************************\n');
    end

    % *************************
    % ** Begin TR Subproblem **
    % *************************
  
    % Determine eta0
    eta = M.zerovec(x);

    %! Solve TR subproblem approximately
    [eta, Heta, numit, stop_inner] = ...
                mtCG(problem, Hx, x, fgradx, eta, Delta, options, storedb, key);
    srstr = tcg_stop_reason{stop_inner};
    
    % This is computed for logging purposes and may be useful for some
    % user-defined stopping criteria.
    % norm_eta = M.norm(x, eta);
    norm_eta = 0;
    
    if options.debug > 0
        testangle = M.inner(x, eta, fgradx) / (norm_eta*norm_grad);
    end
    

    % Compute the tentative next iterate (the proposal)
    x_prop  = problem.R(x, eta);
    key_prop = storedb.getNewKey();

    % Compute the function value of the proposal
    fx_prop = getCost(problem, x_prop, storedb, key_prop);

    % Will we accept the proposal or not?
    % Check the performance of the quadratic model against the actual cost.
    rhonum = fx - fx_prop;
    vecrho = M.lincomb(x, 1, fgradx, .5, Heta);
    rhoden = -M.inner(x, eta, vecrho);
    % rhonum could be anything.
    % rhoden should be nonnegative, as guaranteed by tCG, baring numerical
    % errors.
    
    rho_reg = max(1, abs(fx)) * eps * options.rho_regularization;
    rhonum = rhonum + rho_reg;
    rhoden = rhoden + rho_reg;
   
    if options.debug > 0
        fprintf('DBG:     rhonum : %e\n', rhonum);
        fprintf('DBG:     rhoden : %e\n', rhoden);
    end

    model_decreased = (rhoden >= 0);
    % model_decreased = (rhoden >= 0) && (stop_inner ~= 6);
    
    if ~model_decreased
        srstr = [srstr ', model did not decrease']; %#ok<AGROW>
    end
    
    rho = rhonum / rhoden;
    
    % Added June 30, 2015 following observation by BM.
    % With this modification, it is guaranteed that a step rejection is
    % always accompanied by a TR reduction. This prevents stagnation in
    % this "corner case" (NaN's really aren't supposed to occur, but it's
    % nice if we can handle them nonetheless).
    if isnan(rho)
        fprintf('rho is NaN! Forcing a radius decrease. This should not happen.\n');
        if isnan(fx_prop)
            fprintf('The cost function returned NaN (perhaps the retraction returned a bad point?)\n');
        else
            fprintf('The cost function did not return a NaN value.\n');
        end
    end
   
    if options.debug > 0
        m = @(x, eta) ...
          getCost(problem, x, storedb, key) + ...
          getDirectionalDerivative(problem, x, eta, storedb, key) + ...
             .5*M.inner(x, getHessian(problem, x, eta, storedb, key), eta);
        zerovec = M.zerovec(x);
        actrho = (fx - fx_prop) / (m(x, zerovec) - m(x, eta));
        fprintf('DBG:   new f(x) : %+e\n', fx_prop);
        fprintf('DBG: actual rho : %e\n', actrho);
        fprintf('DBG:   used rho : %e\n', rho);
    end

    % Choose the new TR radius based on the model performance
    trstr = '   ';
    % If the actual decrease is smaller than 1/4 of the predicted decrease,
    % then reduce the TR radius.
    if rho < 1/4 || ~model_decreased || isnan(rho)
        trstr = 'TR-';
        Delta = Delta/4;
        consecutive_TRplus = 0;
        consecutive_TRminus = consecutive_TRminus + 1;
        if consecutive_TRminus >= 5 && options.verbosity >= 2
            consecutive_TRminus = -inf;
            fprintf(' +++ Detected many consecutive TR- (radius decreases).\n');
            fprintf(' +++ Consider decreasing options.Delta_bar by an order of magnitude.\n');
            fprintf(' +++ Current values: options.Delta_bar = %g and options.Delta0 = %g.\n', options.Delta_bar, options.Delta0);
        end
    % If the actual decrease is at least 3/4 of the predicted decrease and
    % the tCG (inner solve) hit the TR boundary, increase the TR radius.
    % We also keep track of the number of consecutive trust-region radius
    % increases. If there are many, this may indicate the need to adapt the
    % initial and maximum radii.
    elseif rho > 3/4 && (stop_inner == 1 || stop_inner == 2)
        trstr = 'TR+';
        Delta = min(2*Delta, options.Delta_bar);
        consecutive_TRminus = 0;
        consecutive_TRplus = consecutive_TRplus + 1;
        if consecutive_TRplus >= 5 && options.verbosity >= 1
            consecutive_TRplus = -inf;
            fprintf(' +++ Detected many consecutive TR+ (radius increases).\n');
            fprintf(' +++ Consider increasing options.Delta_bar by an order of magnitude.\n');
            fprintf(' +++ Current values: options.Delta_bar = %g and options.Delta0 = %g.\n', options.Delta_bar, options.Delta0);
        end
    else
        % Otherwise, keep the TR radius constant.
        consecutive_TRplus = 0;
        consecutive_TRminus = 0;
    end

    % Choose to accept or reject the proposed step based on the model
    % performance. Note the strict inequality.
    if model_decreased && rho > options.rho_prime
        
        % April 17, 2018: a side effect of rho_regularization > 0 is that
        % it can happen that the cost function appears to go up (although
        % only by a small amount) for some accepted steps. We decide to
        % accept this because, numerically, computing the difference
        % between fx_prop and fx is more difficult than computing the
        % improvement in the model, because fx_prop and fx are on the same
        % order of magnitude yet are separated by a very small gap near
        % convergence, whereas the model improvement is computed as a sum
        % of two small terms. As a result, the step which seems bad may
        % turn out to be good, in that it may help reduce the gradient norm
        % for example. This update merely informs the user of this event.
        % In further updates, we could also introduce this as a stopping
        % criterion. It is then important to choose wisely which of x or
        % x_prop should be returned (perhaps the one with smallest
        % gradient?)
        if fx_prop > fx && options.verbosity >= 2
            fprintf(['Between line above and below, cost function ' ...
                     'increased by %s (step size: %g)\n'], ...
                     fx_prop - fx, norm_eta);
        end
        
        accept = true;
        accstr = 'acc';
        % We accept the step: no need to keep the old cache.
        storedb.removefirstifdifferent(key, key_prop);
        x = x_prop;
        key = key_prop;
        fx = fx_prop;
        fgradx = getGradient(problem, x, storedb, key);
        Hx_data = H(x);
        Hx = @(eta) st_proj(x,Hx_data{1} * eta + Hx_data{2} .* eta + eta * Hx_data{3});
        norm_grad = M.norm(x, fgradx);
    else
        % We reject the step: no need to keep cache related to the
        % tentative step.
        storedb.removefirstifdifferent(key_prop, key);
        accept = false;
        accstr = 'REJ';
    end
    
    % k is the number of iterations we have accomplished.
    k = k + 1;
    
    % Make sure we don't use too much memory for the store database
    storedb.purge();
    

    % Log statistics for freshly executed iteration.
    % Everything after this in the loop is not accounted for in the timing.
    stats = savestats(problem, x, storedb, key, options, k, fx, ...
                      norm_grad, Delta, ticstart, info, rho, rhonum, ...
                      rhoden, accept, numit, norm_eta, used_cauchy);
    info(k+1) = stats;

    
    % ** Display:
    if options.verbosity == 2
        fprintf(['%3s %3s   k: %5d     num_inner: %5d     ', ...
        'f: %+e   |grad|: %e   %s\n'], ...
        accstr,trstr,k,numit,fx,norm_grad,srstr);
    elseif options.verbosity > 2
        fprintf('%3s %3s    k: %5d     num_inner: %5d     %s\n', ...
                accstr, trstr, k, numit, srstr);
        fprintf('       f(x) : %+e     |grad| : %e\n',fx,norm_grad);
        if options.debug > 0
            fprintf('      Delta : %f          |eta| : %e\n',Delta,norm_eta);
        end
        fprintf('        rho : %e\n',rho);
    end
    if options.debug > 0
        fprintf('DBG: cos ang(eta,gradf): %d\n',testangle);
        if rho == 0
            fprintf('DBG: rho = 0, this will likely hinder further convergence.\n');
        end
    end

end  % of TR loop (counter: k)

% Restrict info struct-array to useful part
info = info(1:k+1);


if (options.verbosity > 2) || (options.debug > 0)
   fprintf('************************************************************************\n');
end
if (options.verbosity > 0) || (options.debug > 0)
    fprintf('Total time is %f [s] (excludes statsfun)\n', info(end).time);
end

% Return the best cost reached
cost = fx;

end

function Z = st_proj(X,Y)
    tmp = X' * Y;
    Z = Y - 0.5 * X * (tmp + tmp');
end

% Routine in charge of collecting the current iteration stats
function stats = savestats(problem, x, storedb, key, options, k, fx, ...
                           norm_grad, Delta, ticstart, info, rho, rhonum, ...
                           rhoden, accept, numit, norm_eta, used_cauchy)
    stats.iter = k;
    stats.cost = fx;
    stats.gradnorm = norm_grad;
    stats.Delta = Delta;
    if k == 0
        stats.time = toc(ticstart);
        stats.rho = inf;
        stats.rhonum = NaN;
        stats.rhoden = NaN;
        stats.accepted = true;
        stats.numinner = 0;
        stats.stepsize = NaN;
        if options.useRand
            stats.cauchy = false;
        end
    else
        stats.time = info(k).time + toc(ticstart);
        stats.rho = rho;
        stats.rhonum = rhonum;
        stats.rhoden = rhoden;
        stats.accepted = accept;
        stats.numinner = numit;
        stats.stepsize = norm_eta;
        if options.useRand
          stats.cauchy = used_cauchy;
        end
    end
    
    % See comment about statsfun above: the x and store passed to statsfun
    % are that of the most recently accepted point after the iteration
    % fully executed.
    stats = applyStatsfun(problem, x, storedb, key, options, stats);
end
