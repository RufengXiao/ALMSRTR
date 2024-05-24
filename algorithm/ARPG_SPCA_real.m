
function [xoptmain, iter, time, fv, nD, sparsity, avar, time_arr, fs, gxs,totalinneriter, numrestart] = ARPG_SPCA(Xinitial, A, lambda, Ll, Lu, tol, maxiter, Ftol, RPMWHmaxiter, RPMWHtol)
% Min_{X \in St} -Tr(X^T A^T A X) + lambda \|X\|_1

    fhandle = @(x)f(x, A, lambda);
    gfhandle = @(x)gf(x, A, lambda);
    
    [xopt, iter, time, fv, nD, fs, totalinneriter, numrestart,time_arr,gxs] = MPGWH_solver(fhandle, gfhandle, Xinitial, lambda, Ll, Lu, tol, maxiter, Ftol, RPMWHmaxiter, RPMWHtol);
    xopt.main(abs(xopt.main) < 1e-6) = 0;
    sparsity = sum(sum(abs(xopt.main) < 1e-6)) / (prod(size(xopt.main)));
%     fprintf('sparsity:%1.3f\n',sum(sum(abs(xopt.main) < 1e-5)) / (n * p));
    xoptmain = xopt.main;
    % adjusted variance
    [Q, R] = qr(A * xopt.main, 0);
    avar = trace(R * R);
end

function [output, iter, time, fv, err, fs, totalinneriter, numrestart,time_arr,gxs] = MPGWH_solver(fhandle, gfhandle, x0, lambda, Ll, Lu, tol, maxiter, Ftol, RPMWHmaxiter, RPMWHtol)
    delta = 0.0001;
    gamma = 0.5;
%    err = inf;
    gx = inf;
    x1 = x0;
    y1 = x0;
    [fx1, x1] = fhandle(x1);
    gfx1 = gfhandle(x1);
    L0 = Lu;
    L = Ll;
    iter = 0;
    fs(iter + 1) = fx1;
    [n, p] = size(x0.main);
    s1 = 1;
    totalbt = 0;
%     innertol = max(1e-13, min(1e-11,1e-3*sqrt(tol)/(Lu^2)));
    innertol = 1e-15;
    [n, p] = size(x0.main);
    Dxinitial = zeros(p, p);
    Dyinitial = zeros(p, p);
%     innertol = 1e-20;
    gfy1 = gfx1;
    fy1 = fx1;
    fx0 = fx1;
    gfx0 = gfx1;
    fprintf('iter:%d, f:%e\n', iter, fx1);
    N = 5;
    maxN = 5;
    minN = 3;
    justrestart = 0;
    SafeGuardIter = N;
    nDxsq = inf;
    gDx = inf;
    btiter = 0;
    totalinneriter = 0;
    numrestart = 0;
    tt = tic;
    time_arr = zeros(maxiter,1);
    count = 0;
%    while(err > tol && (fx1 > Ftol || count < 2) && iter < maxiter)
    while(gx > tol && (fx1 > Ftol || count < 2) && iter < maxiter)
        % safeguard
        tic;
        if(iter == SafeGuardIter)
            if(justrestart == 1)
                Dx = Dybackup;
                justrestart = 0;
            else
                [Dx, inneriter, Dxinitial] = RPMWH(x0, gfx0, L, lambda, RPMWHtol, false, RPMWHmaxiter, innertol, Dxinitial);
                xc = Dx.y;
                totalinneriter = totalinneriter + inneriter;
%                 xc.main = proximalmap4(x0.main, gfx0 / L, lambda / L);
%                 Dx = Rinv(x0, xc);
            end
            alpha = 1;
            [fxc, xc] = fhandle(xc);
            btiter = 0;
            nDxsq = innerprod(x0, Dx, Dx);
            gDx = max(max(abs(Dx.main)));
            while(fxc > fx0 - delta * alpha * nDxsq && btiter < 3)
                alpha = alpha * gamma;
                newDx.main = alpha * Dx.main;
                xc = RPolar(x0, newDx);
                [fxc, xc] = fhandle(xc);
                btiter = btiter + 1;
                totalbt = totalbt + 1;
            end
            if(btiter == 3)
%                 if(L == L0)
%                     break;
%                 end
                innertol = max(innertol * 1e-2, 1e-24);
%                 innertol = 1e-20;
                L = min(L0, L * 1.1);
                iter = iter + 1;
                fs(iter + 1) = fx1;
                gxs(iter+1) = gx;
                time_arr(iter+1) = toc(tt);
                continue;
            end
            % if safeguard takes effect, here must be a strict inequality,
            % otherwise, algorithm would stack at N = 1.
            if(iter ~= 0 && fxc < fx2)
%                 fprintf('%e, %e\n', fxc, fx2);%%---
                
                gfxc = gfhandle(xc);
                y1 = xc; gfy1 = gfxc; fy1 = fxc;
                x1 = xc; fx1 = fxc; gfx1 = gfxc;
                s1 = 1;
%                 fprintf(['iter: ' num2str(iter) ', restart\n']);
                if(N ~= maxN)
                    L = min(L0, L * 1.1);
                end
                N = max(N - 1, minN);
                justrestart = 1;
                numrestart = numrestart + 1;
            else
                N = min(N + 1, maxN);
            end
            % update x0
            x0 = x1;
            fx0 = fx1;
            gfx0 = gfhandle(x0);
        end
        
        [Dy, inneriter, Dyinitial] = RPMWH(y1, gfy1, L, lambda, RPMWHtol, false, RPMWHmaxiter, innertol, Dyinitial);
        x2 = Dy.y;
%         checkx2 = x2.main' * x2.main%%---
        totalinneriter = totalinneriter + inneriter;
%         x2.main = proximalmap4(y1.main, gfy1 / L, lambda / L);
%         Dy = Rinv(y1, x2);
        if(justrestart == 1 && iter - 1 == SafeGuardIter) % this is to avoid a duplicate computation in the safeguard
            Dybackup = Dy;
        end
        if(iter - 1 == SafeGuardIter) % find the next safeguard iteration.
            SafeGuardIter = SafeGuardIter + N;
        end
        [fx2, x2] = fhandle(x2);
        
        s2 = (1 + sqrt(1 + 4 * s1 * s1)) / 2;
        Rinvy1x1 = InvRPolar(y1, x1);
        zeta.main = (s2 + s1 - 1) / s2 * Dy.main - (s1 - 1) / s2 * Rinvy1x1.main;
        
%         Rinvx2x1 = InvRPolar(x2, x1);
%         zeta.main = (- (s1 - 1) / s2) * Rinvx2x1.main;

        iter = iter + 1;
        fs(iter + 1) = min(fx2, fhandle(x0));
        err = nDxsq * L0 * L0;
        gx = gDx * L0/ (sqrt(sum(sum(x2.main.^2))) + 1) ;
        Ds(iter) = sqrt(nDxsq);
%         fprintf('iter:%d, f:%e, nD^2:%e, inneriterx:%d, inneritery:%d, btiter:%d\n', iter, fx2, err, inneriterx, inneritery, btiter);
        
        y2 = RPolar(y1, zeta);

%         y2 = RPolar(x2, zeta);
        
%         y2.main'*y2.main
%         x2.main'*x2.main
%         y2 = x2;%%---
        
        [fy2, y2] = fhandle(y2);
        gfy2 = gfhandle(y2);
        fy1 = fy2; gfy1 = gfy2;
        y1 = y2;
        s1 = s2;
        x1 = x2; fx1 = fx2;
         if fx1 <= Ftol
            count = count+1;
         end
        
%         if(mod(iter, 10) == 0)
%            fprintf('iter:%d, fx:%e, N:%d, btiter:%d, err:%e, totalinneriter:%d\n', iter, min(fx1, fhandle(x0)), N, btiter, err, totalinneriter);
%             fprintf('iter:%d, fx:%e, N:%d, btiter:%d, optnorm:%e, totalinneriter:%d\n', iter, min(fx1, fhandle(x0)), N, btiter, gx, totalinneriter);
%         end
        gxs(iter+1) = gx;
        time_arr(iter + 1) = toc(tt);
    end
    iter = iter+1;
%    fprintf('iter:%d, fx:%e, err:%e, totalbt:%d, totalinneriter:%d\n', iter, min(fx1, fhandle(x0)), err, totalbt, totalinneriter);
    output = x0;
    fv = fx0;
    time = toc(tt);
        fprintf('iter:%d, fx:%e, optnorm:%e, totalbt:%d, totalinneriter:%d, CPU:%1.2f\n', iter, min(fx1, fhandle(x0)), gx, totalbt, totalinneriter,time);
end

function output = RPolar(x, eta)
% Polar retraction
    [Q,R] = qr(x.main + eta.main,0);
    [U,S,V] = svd(R);
    output.main = Q*(U*V');
end

function output = InvRPolar(x, y)
    A = x.main' * y.main;
    S = lyap(A, -2 * eye(size(A)));
    output.main = y.main * S - x.main;
end

function output = innerprod(x, etax, xix)
    output = etax.main(:)' * xix.main(:);
%     tmp = xix.main - 0.5 * x.main * (x.main' * xix.main);
%     output = etax.main(:)' * tmp(:);
end

function [output, x] = f(x, A, lambda)
    x.Ax = A * x.main;
    tmp = norm(x.Ax, 'fro');
    output = - tmp * tmp + lambda * sum(abs(x.main(:)));
end

function output = gf(x, A, lambda)
    if(~isfield(x, 'Ax'))
        x.Ax = A * x.main;
    end
    gfx = -2 * (A' * x.Ax);
%     output.main = gfx - x.main * (gfx' * x.main);
    tmp = gfx' * x.main;
    output.main = gfx - x.main * ((tmp + tmp') / 2);
end
