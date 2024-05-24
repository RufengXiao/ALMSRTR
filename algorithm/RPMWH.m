
function [etax, iter, Dinitial] = RPMWH(x, gradf, tL, lambda, tol, hasinfo, maxiter, innertol, Dinitial)
% this file is to solve the Riemannian proximal mapping:
% min_{etax} < grad f(x), etax >_x + tL/2 \|etax\|_x^2 + lambda \|R_x(etax)\|_1
% for the Stiefel manifold with the canonical metric
% the retraction is the exponential mapping under the canonical metric
    [n, p] = size(x.main);
    sigma = 0.0001;
%     tol = 1e-3;
    delta = 1;
    fell = @(etax)ell(etax, x, gradf, tL, lambda);
    fprox = @prox;
    fcalJ = @calJ;
    
    fcalA = @calA;
    fcalAstar = @calAstar;
    
%     innertol = max(1e-13, min(1e-11,1e-3*sqrt(tol) / tL / tL));
%     innertol = 1e-20;
    
    % Egradf is the Riemannian gradient under the Euclidean metric
%     Egradf.main = gradf.main - 0.5 * x.main * (x.main' * gradf.main);
    Egradf = gradf;
    
%     Dinitial = zeros(p, p);
    InvAdjOmegaR = zeros(3 * p * p, 1);
    InvOmegaR = zeros(3 * p * p, 1);
    [init_etax.main, Dinitial, inneriter] = finddir(x.main, Egradf.main, delta * tL, fcalA, fcalAstar, fprox, fcalJ, lambda, Dinitial, innertol);

%     zerox.main = zeros(size(init_etax.main));
%     [f1, zerox] = fell(zerox);
%     alpha = 1;
%     dir = init_etax;
%     newetax.main = alpha * dir.main;
%     [f2, newetax] = fell(newetax);
%     btiter = 0;
%     while(f2 > f1 - sigma * alpha * norm(dir.main, 'fro')^2)
%         alpha = alpha * 0.5;
%         newetax.main = alpha * dir.main;
%         [f2, newetax] = fell(newetax);
%         btiter = btiter + 1;
%     end
%     alpha

%     % rescale init_etax and make sure that it is not outside the
%     % diffeomorphism region
%     netax = norm(init_etax, 'fro');
%     init_etax = min(0.1, netax) * init_etax / netax;
    etax = init_etax;
%     etax.main' * x.main%%---
%     Egradfold = Egradf;
    netay = sqrt(innerprod(x, etax, etax)) * delta;
    [f1, etax] = fell(etax);
    iter = 0; err = inf;
    maxbtiter = 20;
    if(hasinfo)
        fprintf('iter:%d, f:%e, |netay|:%e\n', iter, f1, netay);
    end
    tolgmres = tol;
    err = inf;
%     BBstepsize = 1;
%     s = init_etax.main;
    while (netay > tol && iter < maxiter && err > tol)
        y = etax.y;
%         y = RExpStief(x, etax);
        xix.main = gradf.main + tL * etax.main;
%         [tmp1, x, etax] = InvAdjointDiffRPolar(x, etax, xix);
%         Egradf.main = tmp1.main - 0.5 * y.main * (y.main' * tmp1.main);
        [Egradf, x, etax] = InvAdjointDiffRPolar(x, etax, xix);
%         here = Egradf.main
%         Dinitial = zeros(p, p);
        
        [etay.main, Dinitial, inneriter] = finddir(y.main, Egradf.main, delta * tL, fcalA, fcalAstar, fprox, fcalJ, lambda, Dinitial, innertol);
%         h1 = etay' * y
        etay = proj(y, etay);
        netay = sqrt(innerprod(y, etay, etay)) * delta;
%         h2 = etay' * y
        
        % do line search for etay
        alpha = 1;
        [dir, x, etax] = InvDiffRPolar(x, etax, etay);
        dir = proj(x, dir);
        newetax.main = etax.main + alpha * dir.main;
        [f2, newetax] = fell(newetax);
        btiter = 0;
        lsflag = 0;
        while(f2 > f1 - sigma * alpha * norm(dir.main, 'fro')^2 && btiter < maxbtiter)
            lsflag = 1;
            alpha = alpha * 0.5;
            newetax.main = etax.main + alpha * dir.main;
            [f2, newetax] = fell(newetax);
            btiter = btiter + 1;
        end
        if(btiter == maxbtiter)
%             fprintf('warning: RPM line search fails!\n');
            delta = 4 * delta;
            iter = iter + 1;
%             tolgmres = max(netay * 0.01, tol * 0.1);
            if(hasinfo)
                fprintf('iter:%d, f:%e, err:%e, alpha:%e, |etay|:%e, delta:%e, LSfails\n', iter, f1, err, alpha, netay, delta);
            end
            continue;
        end
%         s = alpha * dir.main;
        
        % after line search
        etax = newetax;
        f1 = f2;
        err = norm(alpha * dir.main, 'fro');
        if(lsflag == 1)
            delta = 1.5 * delta;
        else
            delta = delta / 1.5;
        end
        iter = iter + 1;
%         tolgmres = max(netay * 0.01, tol * 0.1);
        if(hasinfo)
            fprintf('iter:%d, f:%e, err:%e, alpha:%e, |etay|:%e, delta:%e\n', iter, f1, err, alpha, netay, delta);
        end
    end
    tmp = x.main' * etax.main;
    etax.main = etax.main - x.main * (tmp' + tmp) / 2;
end

function output = proj(x, z)
    tmp = z.main' * x.main;
    output.main = z.main - x.main * (tmp + tmp') / 2;
end

function [output, x, eta] = RPolar(x, eta)
% Polar retraction
    [Q,R] = qr(x.main + eta.main,0);
    [U,S,V] = svd(R);
    output.main = Q*(U*V');
    eta.y = output;
end

function [output, etax] = ell(etax, x, gradf, tL, lambda)
    [y, x, etax] = RPolar(x, etax);
    output = innerprod(x, gradf, etax) + tL / 2 * innerprod(x, etax, etax) + lambda * norm(y.main(:), 1);
end

function output = innerprod(x, etax, xix)
    output = etax.main(:)' * xix.main(:);
%     tmp = xix.main - 0.5 * x.main * (x.main' * xix.main);
%     output = etax.main(:)' * tmp(:);
end

function [output, x, eta] = InvDiffRPolar(x, eta, Txi)
    y = eta.y; %RPolar(x, eta);
%     tmp1 = (y.main' * (x.main + eta.main));
    tmp1 = eta.tmp1;
    tmp2 = Txi.main * tmp1;
    P = tmp2 - y.main * (y.main' * tmp2);
%     A = x.main' * y.main;
    A = eta.A;
    tmp3 = y.main' * Txi.main;
    tmp4 = x.main' * P;
    Q = (tmp3 * tmp1 + tmp1 * tmp3) * A' - tmp4 - tmp4';
    Omega = lyap(A, -Q);
    output.main = y.main * Omega + P;
end

function [output, x, eta] = InvAdjointDiffRPolar(x, eta, xi)
    y = eta.y; % RPolar(x, eta);
    A = x.main' * y.main;
    Q = y.main' * xi.main;
    B = lyap(A', -Q);
    tmp1 = (y.main' * (x.main + eta.main));
    tmp2 = (x.main * (B + B') - xi.main) * tmp1;
    tmp3.main = y.main * (B * A * tmp1 + tmp1 * B * A) - tmp2 + y.main * (y.main' * tmp2);
    output = proj(y, tmp3);
    eta.A = A;
    eta.tmp1 = tmp1;
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

% Use semi-Newton to solve the following problem:
% < gfx, etax >_E + tL / 2 \|etax\|_E^2 + g(x + etax)
function [output, Lambda, inneriter] = finddir(x, gfx, tL, fcalA, fcalAstar, fprox, fcalJ, mmu, x0, innertol)
    lambda = 0.2;
    nu = 0.99;
    tau = 0.1;
    eta1 = 0.2; eta2 = 0.75;
    gamma1 = 3; gamma2 = 5;
    alpha = 0.1;
    beta = 1 / alpha / 100;
    [n, p] = size(x);
    t = 1 / tL;
    
    z = x0;
%     x
%     gfx
%     fcalAstar(z, x)
    BLambda = x - t * (gfx - fcalAstar(z, x));
    Fz = E(z, BLambda, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu);
    
    nFz = norm(Fz, 'fro');
    nnls = 5;
    xi = zeros(nnls, 1);% for non-monotonic linesearch
    xi(nnls) = nFz;
    maxiter = 1000;
    times = 0;
    Blocks = cell(p, 1);
    while(nFz * nFz > innertol && times < maxiter) % while not converge, find d and update z
        mu = lambda * max(min(nFz, 0.1), 1e-11);
        Axhandle = @(d)GLd(z, d, BLambda, Blocks, x, gfx, t, fcalA, fcalAstar, fprox, fcalJ, mmu) + mu * d;
        [d, CGiter] = myCG(Axhandle, -Fz, tau, lambda * nFz, 30); % update d
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
%         fprintf(['iter:%d, nFz:%e, xi:%f, ' status '\n'], times, nFz, max(xi));
        times = times + 1;
    end
    Lambda = z;
    inneriter = times;
    output = fprox(BLambda, t, mmu) - x;
    
    tmp = output' * x;
    output = output - x * (tmp + tmp') / 2;
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

function [output, k] = myCG(Axhandle, b, tau, lambdanFz, maxiter)
    x = zeros(size(b));
    r = b;
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
    output = x;
end
