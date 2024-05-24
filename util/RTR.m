function [x, fx, out] = RTR(problem, x, opts, varargin)

% Riemannian Trust-Region method with Truncate CG
%   min F(x), s.t., x in M
%
% Input:
%           x --- initial guess
%           f --- objective function
%           g --- Riemann gradient of the objective function
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= OptStiefelGBB(X0, @fun, opts, data1, data2);
%           H --- (general/approximated) Hessian
%           R --- retraction
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 maxiter       max number of iterations
%                 tolgradnorm   tolerance of gradient norm
%                 Delta0      initial region size
%                 Delta_bar   region size upper bound
%                 rho_prime   in [0, 1/4)
%                 theta, kappa       parameters in TCG
%   
% Output:
%           x --- solution
%           f --- function value at x
%         out --- output information
% -----------------------------------------------------------------------

if nargin < 2
    opts = [];
end

% default values
if ~isfield(opts, 'maxiter');      opts.maxiter = 100;  end
if ~isfield(opts, 'Delta_bar');  opts.Delta_bar = 1;  end
if ~isfield(opts, 'Delta0');     opts.Delta0 = opts.Delta_bar / 8;  end
if ~isfield(opts, 'rho_prime');  opts.rho_prime = 0.1;  end
if ~isfield(opts, 'tolgradnorm'); opts.tolgradnorm = 1e-6; end

% fid = 1;
% if opts.record
%     if isfield(opts, 'record_fid')
%         fid = opts.record_fid;
%     elseif isfield(opts, 'recordFile')
%         fid = fopen(opts.recordFile,'w+');
%     end
% end

% copy parameters
costgrad = problem.costgrad;
H = problem.H;
R = problem.R;
maxiter = opts.maxiter;
Delta0 = opts.Delta0;
Delta_bar = opts.Delta_bar;
rho_prime = opts.rho_prime;
tolgradnorm = opts.tolgradnorm;

%% Print iteration header if debug == 1
% if opts.record
%     fprintf(fid,'\n%6s %15s %15s  %16s %9s %9s %5s %6s\n', ...
%         'Iter', 'f(X)', 'Cval', 'nrmG', 'XDiff', 'FDiff', 'nls', 'alpha');
% end
% 
% if record == 10; out.fvec = f; end
% out.msg = 'max_it';

% initial values
Deltak = Delta0;
x_seq = {x};
[fx, gx] = costgrad(x);
f_seq = [fx];
Hx_data = H(x);
Hx = @(eta) Hx_data{1} * eta + Hx_data{2} .* eta + eta * Hx_data{3};

% loop
for iter = 1:maxiter
    eta = TCG(gx, Hx, Deltak);
    x_cand = R(x,eta);
    [f_cand, g_cand] = costgrad(x_cand);
    rho = (fx - f_cand) / m_diff(gx,Hx,eta);

    if rho < 1/4
        Deltak = 1/4 * Deltak;
    elseif rho > 3/4 && iprod(eta,eta) == Deltak^2
        Deltak = min(2 * Deltak, Delta_bar);
    end

    if rho > rho_prime
        x = x_cand;
        x_seq{end+1} = x;
        fx = f_cand;
        gx = g_cand;
        Hx_data = H(x);
        Hx = @(eta) Hx_data{1} * eta + Hx_data{2} .* eta + eta * Hx_data{3};
    else
        x_seq{end+1} = x;
    end

    f_seq(end+1) = fx;
    
    gradnorm = sqrt(iprod(gx,gx));
    if gradnorm < tolgradnorm
        break
    end
    
    % ---- record ----
%     if opts.record
%         fprintf(fid,...
%             '%6d %20.13e %20.13e %9.2e %9.2e %9.2e %2d %9.2e\n', ...
%             iter, f, Cval, nrmG, XDiff, FDiff, nls, alpha);
%     end
end

% if opts.record && isfield(opts, 'recordFile')
%     fclose(fid);
% end

out.iter = iter;
out.gradnorm = gradnorm;
out.Delta = Deltak;
out.x_seq = x_seq;
out.f_seq = f_seq;

% Euclidean inner product
function a = iprod(x,y)
  a = real(sum(sum(conj(x).*y)));
end

% model function
function diff = m_diff(grad, H, eta)
    diff = -1 * iprod(grad,eta) - 0.5 * iprod(eta, H(eta));
end
end
