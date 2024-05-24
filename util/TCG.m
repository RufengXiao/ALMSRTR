function [eta, out] = TCG(grad, H, Delta, opts)

    % Truncate CG
    %
    % Input:
    %           grad --- grad(f(x))
    %           Delta --- radius of the trust region
    %           H --- Hessian operator
    %
    %        opts --- option structure with fields:
    %                 record = 0, no print out
    %                 maxit       max number of iterations
    %                 theta       stop control
    %                 kappa       stop control
    %   
    % Output:
    %           eta --- solution
    %         out --- output information
    % -----------------------------------------------------------------------
    
    if nargin < 3
        error('at least two inputs: [eta, out] = TCG(grad, Delta, opts)');
    elseif nargin < 4
        opts = [];
    end
    
    % termination rule
    if ~isfield(opts, 'theta');      opts.theta = 1;  end
    if ~isfield(opts, 'kappa');      opts.kappa = 1e-6;  end
    if ~isfield(opts, 'maxiter');     opts.maxiter = 100; end

    if ~isa(H,'function_handle')
        H = @(eta) H * eta;
    end
    
%     fid = 1;
%     if opts.record
%         if isfield(opts, 'record_fid')
%             fid = opts.record_fid;
%         elseif isfield(opts, 'recordFile')
%             fid = fopen(opts.recordFile,'w+');
%         end
%     end
    
    % copy parameters
    theta = opts.theta;
    kappa = opts.kappa;
    maxit = opts.maxiter;
%     record = opts.record;
    D2 = Delta^2;
    
    %% Print iteration header if debug == 1
    % if opts.record
    %     fprintf(fid,'\n%6s %15s %15s  %16s %9s %9s %5s %6s\n', ...
    %         'Iter', 'f(X)', 'Cval', 'nrmG', 'XDiff', 'FDiff', 'nls', 'alpha');
    % end
    
    % if record == 10; out.fvec = f; end
    % out.msg = 'max_it';

    % Inital values
    eta = 0;
    ee = 0;
    de = 0;
    r0 = grad; r = r0;
    rr0 = iprod(r0,r0); rr = rr0;
    delta = -r;
    dd = rr;
    trunc = []; % -1: negative curvature; 0: interior; 1: exceed 
    
    % loop
    for iter = 1:maxit
        Hd = H(delta);
        dHd = iprod(delta, Hd);
        if dHd <= 0
            eta = eta + tau(de, dd, ee, D2) * delta;
            trunc(iter) = -1;
            break
        end
        
        alpha = rr / dHd;
        de = iprod(eta,delta);
        dd = iprod(delta,delta);
        % pretend we update eta here
        ee_new = ee + 2 * alpha * de + alpha^2 * dd;

        if ee_new >= D2
            eta = eta + tau(de, dd, ee, D2) * delta;
            trunc(iter) = 1;
            break
        end

        ee = ee_new;
        % actually update eta here
        eta = eta + alpha * delta; 
        trunc(iter) = 0;

        r = r + alpha * Hd;
        rr1 = iprod(r,r);

        if rr1 <= rr0 * min(rr0^(2*theta), kappa^2)
            break
        end
        
        delta = -r + rr1 / rr * delta;
        rr = rr1;
    end
        
        % ---- record ----
%         if opts.record
%             fprintf(fid,...
%                 '%6d %20.13e %20.13e %9.2e %9.2e %9.2e %2d %9.2e\n', ...
%                 iter, f, Cval, nrmG, XDiff, FDiff, nls, alpha);
%         end

 
%     if opts.record && isfield(opts, 'recordFile')
%         fclose(fid);
%     end
    out.trunc = trunc;
    out.iter = iter;
    
    % Euclidean inner product
    function a = iprod(x,y)
      a = real(sum(sum(conj(x).*y)));
    end

    % Compute tau
    function t = tau(de, dd, ee, D2)
        t = (-de + sqrt(de^2 + (D2 - ee) * dd)) / dd;
    end
end