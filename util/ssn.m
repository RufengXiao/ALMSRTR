function [p,out] = ssn(problem, x, options)
    % H is a function handle
    % phi is an objective value function handle
    % R is a retraction function handle
    % phip is phi(p_k)
    % X is a function handle
    % 文章里的p我设为q了，linesearch1里面的

    if nargin < 3
        options = struct();
    end

    if ~isfield(options, 'nu'); options.nu = 1; end
    if ~isfield(options, 'mu'); options.mu = 0.1; end
    if ~isfield(options, 'delta'); options.delta = 0.99; end
    if ~isfield(options, 'mmax'); options.mmax = 1000; end
    if ~isfield(options, 'q'); options.q = 0.05; end
    if ~isfield(options, 'beta0'); options.beta0 = 1; end
    if ~isfield(options, 'beta1'); options.beta1 = 1e-3; end
    if ~isfield(options, 'eta_ratio'); options.eta_ratio = 0.7; end
    if ~isfield(options, 'lstype'); options.lstype = 1; end
    if ~isfield(options, 'maxiter'); options.maxiter = 20; end
    if ~isfield(options, 'tolgradnorm'); options.tolgradnorm = 1e-6; end

    costgrad = problem.costgrad;
    Hdata = problem.H;
    R = problem.R;
    nu = options.nu;
    mu = options.mu;
    delta = options.delta;
    mmax = options.mmax;
    q = options.q;
    beta0 = options.beta0;
    beta1 = options.beta1;
    eta_ratio = options.eta_ratio;
    lstype = options.lstype;
    tolgradnorm = options.tolgradnorm;
 
    %% problem setting
    % nu
    % mu
    % delta
    % mmax
    % q
    % beta0
    % beta1
    
    % if H is not positive definite
    %     V = -Xp;
    % else
    % 要是不能判断的话，可以在后面的计算中若出现<p,H*p>负数则退出循环即可
    
    for k = 1:options.maxiter
        [phip, Xp] = costgrad(x);
        Hdata_x = Hdata(x);
        H = @(V) st_proj(x,Hdata_x{1} * V + Hdata_x{2} .* V + V * Hdata_x{3});

        eta = eta_ratio^k;
        Xpnorm = norm(Xp, 'fro');
        w = Xpnorm^nu;
        eta = min(eta, Xpnorm^(1 + nu));
        [m, n] = size(Xp);

        %% CG method to solve (H+wI)V=(-Xp)
        V = ones(m, n);
        r = H(V) + w * V + Xp;
        p = -r;
        CG_STEP = 0;
        rnorm = norm(r, 'fro');

        while rnorm > eta
            pHp = iprod(p, H(p)); %这里若是负数则退出循环

            if pHp <= 0
                V = -Xp;
                break
            end

            alpha = rnorm^2 / (pHp + w * norm(p, 'fro')^2);
            V = V + alpha * p;
            tmprnorm = rnorm^2;
            r = r + alpha * (H(p) + w * p);
            rnorm = norm(r, 'fro');

            if rnorm <= eta
                break;
            end

            beta = rnorm^2 / tmprnorm;
            p = -r + beta * p;
            CG_STEP = CG_STEP + 1;
        end

        %% linesearch
        if lstype == 1
            % linesearch method 1
            Vnorm = norm(V, 'fro');

            if iprod(-Xp, V) >= min(beta0, beta1 * Vnorm^q) * Vnorm^2
                V = -Xp;
            end

            XpV = iprod(Xp, V);
            % 这里文章里没有说找不到时取不取最大的，不知道是不是一定能找到
            for m = 1:mmax
                [phi_cand, X_cand] = costgrad(R(x,delta^m * V));

                if phi_cand <= phip + mu * delta^m * XpV
                    break
                end
            end
            out.gradnorm = norm(X_cand,'fro');
        else
            % linesearch method 2
            for m = 1:mmax
                [~, X_cand] = costgrad(R(x,delta^m * V));
                norm_X_cand = norm(X_cand, 'fro');
                if norm_X_cand <= (1 - 2 * mu * delta^m) * Xpnorm
                    break
                end
            end
            out.gradnorm = norm_X_cand;                    
        end

        x = R(x,delta^m * V);

        %% Termination condition
        if out.gradnorm <= tolgradnorm
            break
        end
    end

    out.iter = k;
    out.options = options;

    function Z = st_proj(X,Y)
        tmp = X' * Y;
        Z = Y - 0.5 * X * (tmp + tmp');
    end

    function a = iprod(x,y)
        a = real(sum(sum(conj(x).*y)));
    end
end
