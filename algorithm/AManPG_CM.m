
function [xopt, iter, comtime, fv, nD, sparsity, avar, time_arr,fs,gxs] = AManPG_CM(x0, H, mu, L, tol, maxiter, Ftol)
    % parameters for line search
    delta = 0.0001;
    gamma = 0.5;
    [m, n] = size(H);
    p = size(x0.main, 2);
    
    % functions for the optimization problem
    fhandle = @(x)f(x, H, mu);
    gfhandle = @(x)gf(x, H, mu);
    fprox = @prox;
    fcalJ = @calJ;
    
    % functions for the manifold
    fcalA = @calA;
    fcalAstar = @calAstar;
    
    xinitial = x0;
    [xopt, fs, Ds,gxs, time_arr,comtime] = solver(fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, xinitial, L, tol, delta, gamma, maxiter, mu, Ftol);
    xopt.main(abs(xopt.main) < 1e-6) = 0;
    sparsity = sum(sum(abs(xopt.main) < 1e-6)) / (n * p);
    xopt = xopt.main;
    iter = length(fs);
    fv = fs(end);
    nD = Ds(end);
%     fprintf('sparsity:%1.3f\n',sum(sum(abs(xopt.main) < 1e-5)) / (n * p));

    % adjusted variance
    [Q, R] = qr(H * xopt, 0);
    avar = trace(R * R);
end

function [xopt, fs, Ds,gxs, time_arr, comtime] = solver(fhandle, gfhandle, fcalA, fcalAstar, fprox, fcalJ, x0, L, tol, delta, gamma, maxiter, mu, Ftol)
    tt = tic;
%    err = inf;
    gx = inf;
    [fx1, x1] = fhandle(x0);
    gfx1 = gfhandle(x1);
    t = 1 / L;
    t0 = t;
    iter = 0;
    fs(iter + 1) = fx1;
    time_arr = zeros(maxiter,1);
    [n, p] = size(x0.main);
    zeta = zeros(n, p); % this is R_{x_k}^{-1}(y_k)
    s1 = 1;
    Dyinitial = zeros(p, p);
    Dxinitial = zeros(p, p);
    totalbt = 0;
    innertol = max(1e-13, min(1e-11,1e-3*sqrt(tol)*t^2));
%     innertol = 1e-20;
    fprintf('iter:%d, f:%e\n', iter, fx1);
    x1 = x0;
    y1 = x0;
    gfy1 = gfx1;
    fy1 = fx1;
    fx0 = fx1;
    gfx0 = gfx1;
    count = 0;
%    while(err > tol && (fx1 > Ftol || count < 2) && iter < maxiter)
    while(gx > tol && (fx1 > Ftol || count < 2) && iter < maxiter)
        % safeguard
        if(mod(iter, 5) == 0)
            [Dx, Dxinitial, inneriterx] = finddir(iter, x0, gfx0, t, fcalA, fcalAstar, fprox, fcalJ, mu, Dxinitial, innertol);
            alpha = 1;
            xc = R(x0, alpha * Dx);
            [fxc, xc] = fhandle(xc);
            btiter = 0;
            while(fxc > fx0 - delta * alpha * norm(Dx, 'fro')^2 && btiter < 3)
                alpha = alpha * gamma;
                xc = R(x0, alpha * Dx);
                [fxc, xc] = fhandle(xc);
                btiter = btiter + 1;
                totalbt = totalbt + 1;
            end
            if(btiter == 3)
                innertol = max(innertol * 1e-2, 1e-20);
                iter = iter + 1;
                fs(iter + 1) = fx1;
                gxs(iter+1) = gx;
                time_arr(iter+1) = toc(tt);
                continue;
            end
            if(iter ~= 0 && fxc < fx2) % if safeguard takes effect
                gfxc = gfhandle(xc);
                y1 = xc;gfy1 = gfxc; fy1 = fxc;
                x1 = xc;
                s1 = 1;
%                 fprintf('restart\n');
            end
            % update x0
            x0 = x1;
            fx0 = fhandle(x0);
            gfx0 = gfhandle(x0);
        end
        [Dy, Dyinitial, inneritery] = finddir(iter, y1, gfy1, t, fcalA, fcalAstar, fprox, fcalJ, mu, Dyinitial, innertol);
        alpha = 1;
        x2 = R(y1, alpha * Dy);
        [fx2, x2] = fhandle(x2);
        
        s2 = (1 + sqrt(1 + 4 * s1 * s1)) / 2;
        zeta = (- (s1 - 1) / s2) * Rinv(x2, x1);

        iter = iter + 1;
%        err = (norm(Dx, 'fro') / t)^2;
        gx = max(max(abs(Dx))) / (sqrt(sum(sum(x2.main.^2))) + 1) / t;
        Ds(iter) = norm(Dx, 'fro');
%         if(mod(iter, 10) == 0)
% %            fprintf('iter:%d, f:%e, err:%e, inneriterx:%d, inneritery:%d, btiter:%d\n', iter, fx2, err, inneriterx, inneritery, btiter);
%             fprintf('iter:%d, f:%e, optnorm:%e, inneriterx:%d, inneritery:%d, btiter:%d\n', iter, fx2, gx, inneriterx, inneritery, btiter);
%         end
        
        y2 = R(x2, zeta);
        [fy2, y2] = fhandle(y2);
        gfy2 = gfhandle(y2);
        fy1 = fy2; gfy1 = gfy2;
        y1 = y2;
        s1 = s2;
        x1 = x2; fx1 = fx2;
        fs(iter + 1) = fx1;
        gxs(iter+1) = gx;
        time_arr(iter+1) = toc(tt);
        if fx1 <= Ftol
            count = count+1;
        end
    end

    xopt = x2;
    comtime = toc(tt);
    %    fprintf('iter:%d, f:%e, err:%e,totalbt:%d\n', iter, fx2, err, totalbt);
    fprintf('iter:%d, f:%e, optnorm:%e,totalbt:%d, CPU:%1.2f\n', iter, fx2, gx, totalbt,comtime);
end

function output = proj(x, eta)
    tmp = x.main' * eta;
    output = eta - x.main * (tmp + tmp')/2;
end

function output = Rinv(x, y)
    A = x.main' * y.main;
    S = lyap(A, -2 * eye(size(A)));
    output = y.main * S - x.main;
end

function output = R(x, eta)
    [Q,R] = qr(x.main + eta,0);
    [U,S,V] = svd(R);
    output.main = Q*(U*V');
end

function [output, x] = f(x, H, mu)
    x.Ax = H * x.main;
    tmp = -sum(sum(x.main.*(x.Ax)));
    output = tmp + mu * sum(abs(x.main(:)));
end

function output = gf(x, H, mu)
    if(~isfield(x, 'Ax'))
        x.Ax = H * x.main;
    end
    gfx = -2 * x.Ax;
    tmp = gfx' * x.main;
    output = gfx - x.main * ((tmp + tmp') / 2);
end

% compute E(Lambda)
function ELambda = E(Lambda, BLambda, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu)
    if(length(BLambda) == 0)
        BLambda = x - t * (gfx - fcalAstar(Lambda, x));
    end
    DLambda = fprox(BLambda, t, mmu) - x;
    ELambda = fcalA(DLambda, x);
end

% compute calG(Lambda)[d]
function GLambdad = GLd(Lambda, d, BLambda, Blocks, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu)
        GLambdad = t * fcalA(fcalJ(BLambda, fcalAstar(d, x), t, mmu), x);
end

% Use semi-Newton to solve the subproblem and find the search direction
function [output, Lambda, inneriter] = finddir(iter, xx, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu, x0, innertol)
    x = xx.main;
    lambda = 0.2;
    nu = 0.9999;
    tau = 0.1;
    eta1 = 0.2; eta2 = 0.75;
    gamma1 = 0.1; gamma2 = 5;
    alpha = 0.1;
    beta = 1 / alpha / 100;
    [n, p] = size(x);
    
    z = x0;
    BLambda = x - t * (gfx - fcalAstar(z, x));
    Fz = E(z, BLambda, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
    
    nFz = norm(Fz, 'fro');
    nnls = 5;
    xi = zeros(nnls, 1);% for non-monotonic linesearch
    xi(nnls) = nFz;
    maxiter = 100;
    times = 0;
    Blocks = cell(p, 1);
    while(nFz * nFz > innertol && times < maxiter) % while not converge, find d and update z
        mu = lambda * max(min(nFz, 0.1), 1e-11);
        Axhandle = @(d)GLd(z, d, BLambda, Blocks, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu) + mu * d;
        [d, CGiter, nr] = myCG(Axhandle, -Fz, tau, lambda * nFz, 30); % update d
        u = z + d;
        Fu = E(u, [], x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu); 
        nFu = norm(Fu, 'fro');

        if(nFu < nu * max(xi))
            z = u;
            Fz = Fu;
            nFz = nFu;
            xi(mod(times, nnls) + 1) = nFz;
            status = 'success';
        else
            rho = - sum(Fu(:) .* d(:)) / norm(d, 'fro')^2;
            if(rho >= eta1)
                v = z - sum(sum(Fu .* (z - u))) / nFu^2 * Fu;
                Fv = E(v, [], x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
                nFv = norm(Fv, 'fro');
                if(nFv <= nFz)
                    z = v;
                    Fz = Fv;
                    nFz = nFv;
                    status = 'safegard success projection';
                else
                    z = z - beta * Fz;
                    Fz = E(z, [], x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
                    nFz = norm(Fz, 'fro');
                    status = 'safegard success fixed-point';
                end
            else
%                 fprintf('unsuccessful step\n');
                status = 'safegard unsuccess';
            end
            if(rho >= eta2)
                lambda = max(lambda / 4, 1e-5);
            elseif(rho >= eta1)
                lambda = (1 + gamma1) / 2 * lambda;
            else
                lambda = (gamma1 + gamma2) / 2 * lambda;
            end
        end
        BLambda = x - t * (gfx - fcalAstar(z, x));
%         fprintf(['iter:%d, nFz:%e, lambda:%e, mu:%e, nFu:%e, xi:%e, ' status '\n'], times, nFz, lambda, mu, nFu, max(xi));
        times = times + 1;
    end
    Lambda = z;
    inneriter = times;
    output = fprox(BLambda, t, mmu) - x;
end

function output = prox(X, t, mu)
    output = min(0, X + t * mu) + max(0, X - t * mu);
end

function output = calA(Z, U) % U \in St(p, n)
    tmp = Z' * U;
    output = tmp + tmp';
end

function output = calAstar(Lambda, U) % U \in St(p, n)
    output = U * (Lambda + Lambda');
end

function output = calJ(y, eta, t, mu)
    output = (abs(y) > mu * t) .* eta;
end

function [output, k, nr] = myCG(Axhandle, b, tau, lambdanFz, maxiter)
    x = zeros(size(b));
    r = b;% - Axhandle(x);
    p = r;
    k = 0;
    while(norm(r, 'fro') > tau * min(lambdanFz * norm(x, 'fro'), 1) && k < maxiter)
        Ap = Axhandle(p);
        alpha = r(:)' * r(:) / (p(:)' * Ap(:));
        x = x + alpha * p;
        rr0 = r(:)' * r(:);
        r = r - alpha * Ap;
        beta = r(:)' * r(:) / rr0;
        p = r + beta * p;
        k = k + 1;
    end
    nr = norm(r, 'fro');
    output = x;
end
