% Use semi-Newton to solve the subproblem and find the search direction
function [output, Lambda, inneriter] = mialm_finddir(x, A, mmu)
    [n, p] = size(x);
    [~, S, ~] = svd(A, 'econ');
    D = diag(S(1:p, 1:p));
    Dsq = D.^2;
    L = max(Dsq) * 2;
    tol = 1e-16;
    xx.main = x;
    gfx = gf(xx, A, mmu);
    t = 1 / L;
    fcalA = @calA;
    fcalAstar = @calAstar;
    fprox = @prox;
    fcalJ = @calJ;
    x0 = zeros(p, p);
    innertol = max(1e-13, min(1e-11,1e-3*sqrt(tol)*t^2));
    
    lambda = 0.2;
    nu = 0.9999;
    tau = 0.1;
    eta1 = 0.2; eta2 = 0.75;
    gamma1 = 0.1; gamma2 = 5;
    alpha = 0.1;
    beta = 1 / alpha / 100;
    
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

function output = gf(x, A, mu)
    if(~isfield(x, 'Ax'))
        x.Ax = A * x.main;
    end
    gfx = -2 * (A' * x.Ax);
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
