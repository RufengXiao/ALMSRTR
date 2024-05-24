classdef CMRTR
    %SPCARTR to solve SPCA
    % U denotes lambda
    
    properties
        U  % denotes lambda for convenience
        X  % primal varaiable
        Y  % auxiliary varaiable
        n  % X is a nxr matrix
        r
        sigma % Lagragian penalty parameter
        mu % parameter for l1_norm function of the objective
        H % Data matrix of the objective
        tau % if delta_{k-1} <= tau * delta_k then sigma stay the same
        Sv % svds(full(A), 1)
        maxtr_iter % maxiter for the tr step to solve the subproblem
        sigma_factor % sigma increase ratio in sigma update
        manifold % to define the mainfold then can use the manopt package
        verbose
        iter_num % counts for outer loop
        residual % max(max(abs(X-Y)))
        retraction % retract to the manifold
        maxtcg_iter % for tcg maxiter
        algname % name for the algorithm
        record % record some information
        elapsed_time % running time
        epsilon % subproblem tolerance
        epsilon_min % the min of epsilon
        delta % the initial radius of TR to solve subproblem
        delta_max % the max radius of TR to solve subproblem
        delta_min % the min radius of TR to solve subproblem
        rho0 % tr update threshold
        tcg_record % record some information for tcg
        eta % eta from tcg
        eta_min % min eta norm^2 
        tcg_normrr_tol % for tcg_normrr tolerance
        zeros_eta
        % following parameter is used to KKT
        gX
        gY
        gU
        sparse
        loss 
        tcg_count % the total count of tcg
        % following parameter is used to safeguard
        minGX
        minGX_ind % record the minGX iter number
        rec_c0 % record gradnorm no decrease situation counts
        rec_c % record worse situation counts
        rec_X % record X when the minGX 
        rec_Y % record Y when the minGX
        rec_U % record U when the minGX
        rec_sigma % record sigma when the minGX
        rec_epsilon
        rec_maxtr_iter
        RGBB_COUNT

        fail_count % count of fail to find the subproblem solution
        rec_fail_X % record X when fail to find the subproblem solution 
        rec_fail_Y % record Y when fail to find the subproblem solution 
        rec_fail_U % record U when fail to find the subproblem solution 
        rec_fail_sigma % record sigma when fail to find the subproblem solution 

        % following parameters is used to observe
        back_situation1_count
        back_situation2_count
        back_situation3_count

    end
    
    methods

    function ret = H_mul(obj, X)
        ret = obj.H * X;
        end
    
        function ret = l1_prox(obj, X, lam)
            ret = sign(X) .* max(0, abs(X) - lam);
        end

        % l1_moreau denote the moreau-envelop of lam*||x||_1 
        function ret = l1_moreau(obj, X, lam)
            p = (abs(X) < lam);
            ret = p .* (0.5 * X .* X) + (1 - p) .* (lam * abs(X) - 0.5 * lam * lam);
            ret = sum(sum(ret));
        end    

        function ret = objective_smooth(obj, X)
        ret = sum(sum(X .* obj.H_mul(X)));
        end
    
        function ret = objective(obj, X)
            ret = obj.objective_smooth(X) + obj.mu * sum(sum(abs(X)));
        end

        
        % alm_cost denote f(x) + sigma*moreau(g/sigma(X+lambda/sigma))
        function ret = alm_cost(obj, X)
            %
            G = obj.sigma * obj.l1_moreau(X + obj.U / obj.sigma, obj.mu / obj.sigma);
            f = obj.objective_smooth(X);
            ret = f + G;
        end
    
        % Euclidean grad for subproblem
        function ret = alm_costgrad(obj, X)
            T = X + obj.U / obj.sigma;
            ret = 2 * obj.H_mul(X) + obj.sigma * (T - obj.l1_prox(T, obj.mu / obj.sigma));
        end
    
        function [F, G] = alm_cost_costgrad(obj, X)
            F = obj.alm_cost(X);
            G = obj.alm_costgrad(X);
        end
    
        function DzG = ehess(obj, X, Z)
            T = X + obj.U / obj.sigma;
            E = (abs(T) <= obj.mu / obj.sigma);
            DzG = 2 * obj.H_mul(Z) + obj.sigma * Z .* E;
        end
    
        function H = rhess(obj, X, Z)
            H = obj.manifold.ehess2rhess(X, obj.alm_costgrad(X), obj.ehess(X, Z), Z);
        end

        function ret = res_cost_tr_subproblem(obj, X, eta, rgrad)
            iprod = @(X,Y) real(sum(sum(conj(X).*Y)));
            ret = -iprod(rgrad,eta)-0.5*iprod(obj.rhess(X,eta),eta);
        end

        function obj = tcg4trsubproblem(obj,X,Delta,rgrad,opts)
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
            iprod = @(X,Y) real(sum(sum(conj(X).*Y)));
            tau = @(de, dd, ee, D2) (-de + sqrt(max(de^2 + (D2 - ee) * dd,0))) / dd;
            % copy parameters
            theta = opts.theta;
            kappa = opts.kappa;
            maxit = opts.maxiter;
            obj.tcg_record.norm_rr = [];
%             normrr_tol = min(obj.tcg_normrr_tol, obj.gradnorm);
            normrr_tol = obj.tcg_normrr_tol;

            D2 = Delta^2;

            % Inital values
            if obj.tcg_record.back_flag == 1
                eta = obj.tcg_record.eta1;
                ee = obj.tcg_record.ee1;
                r0 = rgrad;
                r = obj.tcg_record.r1;
                rr0 = iprod(r0,r0);
                rr = obj.tcg_record.rr1;
                delta = obj.tcg_record.delta1;
                obj.tcg_record.tr_back_count = 1;
                obj.tcg_record.buffer_flag1 = 0;
            elseif obj.tcg_record.back_flag == 2
                eta = obj.tcg_record.eta2;
                ee = obj.tcg_record.ee2;
                r0 = rgrad;
                r = obj.tcg_record.r2;
                rr0 = iprod(r0,r0);
                rr = obj.tcg_record.rr2;
                delta = obj.tcg_record.delta2;
                obj.tcg_record.tr_back_count = 0;
                obj.tcg_record.buffer_flag2 = 0;
            elseif obj.tcg_record.back_flag == 3
                eta = obj.tcg_record.eta3;
                ee = obj.tcg_record.ee3;
                r0 = rgrad;
                r = obj.tcg_record.r3;
                rr0 = iprod(r0,r0);
                rr = obj.tcg_record.rr3;
                delta = obj.tcg_record.delta3;
                obj.tcg_record.tr_back_count = 2;
                obj.tcg_record.buffer_flag1 = 0;
                obj.tcg_record.buffer_flag2 = 0;
            elseif obj.tcg_record.back_flag == 0
                eta = obj.zeros_eta;
                ee = 0;
                r0 = rgrad; r = r0;
                rr0 = iprod(r0,r0); rr = rr0;
                delta = -r;
                obj.tcg_record.tr_back_count = 2;
                obj.tcg_record.buffer_flag1 = 0;
                obj.tcg_record.buffer_flag2 = 0;
            end
            tcg_flag = 0; % -1: negative curvature; 0: interior; 1: exceed radius; 2: too small; 3: exceed maxiter or exceed normrr_tol
            record_flag1 = 1;
            record_flag2 = 1;

            for iter = 1:maxit
                if sqrt(rr) < normrr_tol
                    tcg_flag = 3;
                    break;
                end
                Hd = obj.rhess(X, delta);
                dHd = iprod(delta,Hd);
                de = iprod(eta,delta);
                dd = iprod(delta,delta);
        
                % if dHd < 1e-6*dd
                %     tcg_flag = 2;
                %     break;
                % end
%                 if de^2 + (D2 - ee) * dd <= 0
%                     fprintf("error");
%                 end
                % if <delta_j, H_k delta_j> <= 0 
                if dHd <= 0
                    eta = eta + tau(de, dd, ee, D2) * delta;
                    tcg_flag = -1;
                    break
                end
        
                alpha = rr/dHd;
        
                ee_new = ee + 2 * alpha * de + alpha^2 * dd;

                if obj.tcg_record.tr_back_count == 2
                    if ee_new >= 1/16*D2 && record_flag1
                        obj.tcg_record.eta1 = eta;
                        obj.tcg_record.ee1 = ee;
                        obj.tcg_record.r1 = r;
                        obj.tcg_record.rr1 = rr;
                        obj.tcg_record.delta1 = delta;
                        record_flag1 = 0;
                        obj.tcg_record.buffer_flag1 = 1;
                    end

                    if ee_new >= 1/256*D2 && record_flag2
                        obj.tcg_record.eta2 = eta;
                        obj.tcg_record.ee2 = ee;
                        obj.tcg_record.r2 = r;
                        obj.tcg_record.rr2 = rr;
                        obj.tcg_record.delta2 = delta;
                        record_flag2 = 0;
                        obj.tcg_record.buffer_flag2 = 1;
                    end
                end

        
                % if ||ki|| >= Delta
                if ee_new >= D2
                    eta = eta + tau(de, dd, ee, D2) * delta;
                    tcg_flag = 1;
                    break
                end
        
                ee = ee_new;
                % actually update eta
                eta = eta + alpha * delta;
                
                r = r + alpha * Hd;
                rr1 = iprod(r,r);
        
                % if ||r_new||<=||r_0||*min(||r_0||^theta,kappa)
                if rr1 <= rr0 * min(rr0^(2*theta), kappa^2)
                    break
                end
        
                delta = -r + rr1 / rr * delta;
                rr = rr1;  
                obj.tcg_record.norm_rr = [obj.tcg_record.norm_rr sqrt(rr)];  
            end
            if iter >= maxit
                tcg_flag = 3;
            end
            %% record 
            if obj.tcg_record.tr_back_count == 2
                if tcg_flag == 0 || tcg_flag == 3
                    obj.tcg_record.eta3 = eta;
                    obj.tcg_record.ee3 = ee;
                    obj.tcg_record.r3 = r;
                    obj.tcg_record.rr3 = rr;
                    obj.tcg_record.delta3 = delta;
                else 
                    obj.tcg_record.eta3 = eta - tau(de, dd, ee, D2) * delta;
                    obj.tcg_record.ee3 = ee;
                    obj.tcg_record.r3 = r;
                    obj.tcg_record.rr3 = rr;
                    obj.tcg_record.delta3 = delta;
                end
            end
            
            obj.tcg_record.norm_rr = [obj.tcg_record.norm_rr sqrt(rr)];
            obj.tcg_record.tcg_flag = tcg_flag;
            obj.tcg_record.iter = iter;
            obj.eta = eta;
        end

        function obj = subopt_tr(obj)
            % the cost,grad of the subproblem
            cost = @(X) obj.alm_cost(X);
            g_cost = @(X) obj.alm_costgrad(X);

                X = obj.X;

            rgrad = obj.manifold.proj(X, g_cost(X));
            if obj.verbose
                fprintf("Begin: rgradnorm = %1.7e\n", sqrt(sum(sum(rgrad.^2))));
            end

            phi = cost(X);

            delta = obj.delta;
            obj.tcg_record.back_flag = 0; % back eta type
            obj.tcg_record.tr_back_count = 2;
            obj.tcg_record.buffer_flag1 = 0;
            obj.tcg_record.buffer_flag2 = 0;

            for it = 1:obj.maxtr_iter

                options_tcg.theta = 1;
                options_tcg.kappa = 1e-6;
                options_tcg.maxiter = 100;   

                obj = obj.tcg4trsubproblem(X, delta, rgrad, options_tcg);
                obj.tcg_count = obj.tcg_record.iter + obj.tcg_count;

                % tcg_flag: -1: negative curvature; 0: interior; 1: exceed
                % ;2: small; 3: exceed maxiter
                if obj.verbose == 2
                    fprintf('#tcg counts: %d, |eta|_2 = %1.9e, tcg_flag = %d, norm_rr = %1.9e, radius = %1.2e\n',obj.tcg_record.iter,sqrt(sum(sum(obj.eta.^2))),obj.tcg_record.tcg_flag,obj.tcg_record.norm_rr(end),delta);
                end

                % compute rho
                tempX = obj.retraction(X,obj.eta);
                temp_phi = cost(tempX);
                rho = (phi - temp_phi)/obj.res_cost_tr_subproblem(X,obj.eta,rgrad);

                % update radius
                if rho < 1/4
                    delta = 1/4*delta;

                    if obj.tcg_record.buffer_flag1 &&  obj.tcg_record.buffer_flag2
                        obj.tcg_record.back_flag = 1;
                    elseif ~obj.tcg_record.buffer_flag1 &&  obj.tcg_record.buffer_flag2
                        if obj.tcg_record.tr_back_count == 1
                            obj.tcg_record.back_flag = 2;
                        elseif obj.tcg_record.tr_back_count == 2 && obj.tcg_record.tcg_flag == 3
                            delta = delta * 1/4;
                            obj.tcg_record.back_flag = 2;
                        else
                            obj.tcg_record.back_flag = 3;
                        end
                    elseif obj.tcg_record.buffer_flag1 &&  ~obj.tcg_record.buffer_flag2
                        obj.tcg_record.back_flag = 1;
                    elseif ~obj.tcg_record.buffer_flag1 &&  ~obj.tcg_record.buffer_flag2 && obj.tcg_record.tcg_flag == 3
                        delta =  delta * 1/16;
                        obj.tcg_record.back_flag = 0;
                    elseif ~obj.tcg_record.buffer_flag1 &&  ~obj.tcg_record.buffer_flag2 && obj.tcg_record.tcg_flag ~= 3 && obj.tcg_record.tr_back_count == 1
                        obj.tcg_record.back_flag = 3;
                    else
                        obj.tcg_record.back_flag = 0;
                    end
                elseif rho > 3/4 && abs(sum(sum(obj.eta.^2))-delta^2)<1e-6
                    delta = min(2*delta,obj.delta_max);
                end

                if delta < obj.delta_min
                    break;
                end

                % update x
                if rho > obj.rho0
                    X = tempX;
                    phi = temp_phi;
                    rgrad = obj.manifold.proj(tempX, g_cost(tempX));
                    obj.tcg_record.back_flag = 0;
                    if obj.verbose == 2
                        fprintf('After TR success: |rgrad|_2: %1.9e phi: %1.9e, epsilon: %1.6e, sigma: %f\n',sqrt(sum(sum(rgrad.^2))), phi,obj.epsilon,obj.sigma);
                    end
                    if sqrt(sum(sum(rgrad.^2))) <= obj.epsilon
                        if obj.n>=1500
                            if obj.epsilon < 5e-6
                                obj.maxtr_iter = 20;
                            elseif obj.epsilon < 5e-3
                                obj.maxtr_iter = 25;
                            else
                                obj.maxtr_iter = 30;
                            end
                        end
                        obj.fail_count = 0;
                        obj.epsilon = max(obj.epsilon*0.8, obj.epsilon_min); % 0.8 for 200 %

                        if obj.verbose == 2
                            fprintf('Find the solution for subproblem.')
                        end
                        break;
                    end
                end
            end

            if it > obj.maxtr_iter
                if obj.verbose == 2
                    fprintf('Fail to find the solution for subproblem.')
                end
                obj.fail_count = obj.fail_count + 1;
                end

            if obj.fail_count == 1
                obj.rec_fail_X = obj.X;
                obj.rec_fail_Y = obj.Y;
                obj.rec_fail_U = obj.U;
                obj.rec_fail_sigma = obj.sigma;
            end

            obj.X = X;

        end


        function obj = update(obj)
            obj = obj.subopt_tr();
            obj.Y = obj.l1_prox(obj.X + obj.U / obj.sigma, obj.mu / obj.sigma);
            obj.U = obj.U + obj.sigma * (obj.X - obj.Y);

            % RGBB for robust
%             if obj.iter_num > 20 && (obj.record.gX(end) > 200*(2*obj.RGBB_COUNT^3+10)*obj.minGX || obj.record.gX(end) > obj.record.gU(end)*200*(2*obj.RGBB_COUNT^3+10)) && obj.record.gX(end) > obj.n/1e5
%                 options_sd.maxiter = 300 + 20*obj.RGBB_COUNT;
%                 if obj.RGBB_COUNT == 0
%                     obj.RGBB_COUNT = 1;
%                     obj.X = zeros([obj.n, obj.r]);
%                     obj.X(1:obj.r, 1:obj.r) = eye(obj.r);
%                     obj.sigma = 1;
%                 else
%                     obj.RGBB_COUNT = obj.RGBB_COUNT + 1;
%                     obj.X = obj.rec_X;
%                     obj.Y = obj.rec_Y;
%                     obj.U = obj.rec_U;
%                     if obj.n <= 2000
%                         obj.sigma = obj.rec_sigma;
%                     else
%                         obj.sigma = obj.rec_sigma * 0.15;
%                     end
%                     obj.epsilon = obj.rec_epsilon * 1.5;
%                 end
%                 cost_and_grad = @(X) obj.alm_cost_costgrad(X);
%   
%                 options_sd.tolgradnorm = obj.minGX * obj.n;
%                 options_sd.verbosity = obj.verbose;
%     
%                 options_sd.record = 0;
%                 options_sd.mxitr = options_sd.maxiter;
%                 options_sd.gtol = options_sd.tolgradnorm;
%                 options_sd.xtol = 1.0e-20;
%                 options_sd.ftol = 1.0e-20;
%         
%                 [obj.X, info] = OptStiefelGBB(obj.X, cost_and_grad, options_sd);
%                 
%                 obj.minGX = max((obj.RGBB_COUNT==0)*1e10,obj.minGX*1e2);
%                 if obj.verbose
%                     fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, norm(XT*X-I): %3.2e \n', ...
%                             info.fval, info.itr, info.nfe, norm(obj.X'*obj.X - eye(obj.r), 'fro') );
%                end 
%             end
            obj = obj.KKT();
            residual = max(max(abs(obj.X - obj.Y)));
            L = -0.5 * log10(sum(sum(obj.gX.^2))) + 0.5 * log10(sum(sum(obj.gU.^2)));
            if residual >= obj.tau * obj.residual && L > -1.7 || L > 0.4       
                if obj.epsilon > 5e-6 || obj.sigma < obj.n * 10 
                    if  obj.n >= 1500
                        obj.sigma = obj.sigma * 2;
                    else
                        obj.sigma = obj.sigma * 2.5;
                    end 
                else
                obj.sigma = obj.sigma * obj.sigma_factor;
                end
                obj.sigma = max(obj.sigma, sum(sum(obj.U.^2))^0.51);
            end
%             if obj.sigma > obj.n * 40 && obj.record.gX(end) > 1e-5
%                 obj.sigma = obj.sigma * 0.5;
%             end
    
            obj.residual = residual;
            obj.iter_num = obj.iter_num + 1;
%             if obj.iter_num > 10 && obj.record.gX(end) < obj.minGX
%                 obj.minGX_ind = obj.iter_num;
%                 obj.minGX = obj.record.gX(end);
%                 obj.rec_X = obj.X;
%                 obj.rec_Y = obj.Y;
%                 obj.rec_U = obj.U;
%                 obj.rec_sigma = obj.sigma;
%                 obj.rec_epsilon = obj.epsilon;
%             end

            % ==== safeguard ====
%             if obj.iter_num > 15
%                 if obj.record.gX(end) > obj.minGX * 50 || obj.iter_num - obj.minGX_ind > 10 || L < -2.0
%                     obj.rec_c = obj.rec_c + 1;
%                 end
%                 if obj.record.gX(end) > obj.record.gX(end-1) * 50 && obj.record.gX(end) > obj.record.gX(end-2)
%                     obj.rec_c0 = obj.rec_c0 + 1; % the second increase counts
%                 end

                % back situation 1
%                 if obj.rec_c > 10  
%                     obj.rec_c = 0;
%                     if obj.epsilon < 5e-6
%                         obj.maxtr_iter = min(20,ceil(obj.maxtr_iter * 1.25)); 
%                     elseif obj.epsilon < 5e-3
%                         obj.maxtr_iter = min(40,ceil(obj.maxtr_iter * 1.25));
%                     else
%                         obj.maxtr_iter = min(60,ceil(obj.maxtr_iter * 1.25));
%                     end
%                     if obj.sigma < obj.n * 5 || obj.epsilon > 8e-4
%                         if obj.n > 2000
%                             obj.sigma = obj.sigma * 1.45;
%                         elseif obj.n >= 1000
%                             obj.sigma = obj.sigma * 1.8;
%                         elseif obj.n >= 500
%                             obj.sigma = obj.sigma * 1.8;
%                         else
%                             obj.sigma = obj.sigma * 2.25;
%                         end 
%                     end
% %                     if obj.sigma > obj.n * 40 && obj.record.gX(end) > 1e-5
% %                         obj.sigma = obj.sigma * 0.5;
% %                     end
%                     if obj.verbose
%                         fprintf('back situation 1 --> maxtr_iter = %d\n', obj.maxtr_iter);
%                     end
%                     obj.back_situation1_count = obj.back_situation1_count + 1;
%                 end

%                 % back situation 2
%                 if obj.rec_c0 > 15
%                     obj.rec_c0 = 0;
%                     obj.X = obj.rec_X;
%                     obj.Y = obj.rec_Y;
%                     obj.U = obj.rec_U;
%                     if obj.epsilon < obj.rec_epsilon
%                         obj.sigma = obj.rec_sigma * (1 - 400 / obj.n);
%                     else
%                         obj.sigma = obj.rec_sigma * (1 - 200 / obj.n);
%                     end
%                     obj.rec_sigma = obj.sigma;
%                     obj.maxtr_iter = 150;
%                     obj.epsilon = obj.epsilon / (1 - 300 / obj.n);
%                     if obj.verbose
%                         fprintf('back situation 2 --> maxtr_iter = %d\n', obj.maxtr_iter);
%                     end
%                     obj.back_situation2_count = obj.back_situation2_count + 1;
%                 end
%             end

%             % back situation 3
%             if obj.fail_count > 10
%                 obj.X = obj.rec_X;
%                 obj.Y = obj.rec_Y;
%                 obj.U = obj.rec_U;
%                 obj.sigma =  obj.rec_sigma * (1 - 300 / obj.n);
%                 obj.epsilon = obj.epsilon / (1 - 300 / obj.n);
%                 obj.fail_count = 0;
%                 obj.maxtr_iter = 200; % 150 for <2000
%                 if obj.verbose
%                     fprintf('back situation 3 --> maxtr_iter = %d\n', obj.maxtr_iter);
%                 end
%                 obj.back_situation3_count = obj.back_situation3_count + 1;
%             end
    
        end
    
        function obj = KKT(obj)
            obj.gX = 2 * obj.H_mul(obj.X) + obj.U;
            obj.gX = obj.manifold.proj(obj.X, obj.gX) / (sqrt(sum(sum(obj.X.^2))) + 1);
            zero_Y = (abs(obj.Y) < 1.0e-6);
            obj.gY = (1 - zero_Y) .* (obj.mu * sign(obj.Y) - obj.U) - zero_Y .* obj.l1_prox(obj.U, obj.mu);
            obj.gY = obj.gY / (sqrt(sum(sum(obj.Y.^2))) + 1);
            obj.gU = obj.X - obj.Y;
            obj.sparse = sum(sum(abs(obj.X) < 1.0e-6)) / (obj.n * obj.r);
            obj.loss = obj.objective(obj.X);
            obj.record.gX = [ obj.record.gX max(max(abs(obj.gX))) ]; 
            obj.record.gY = [ obj.record.gY max(max(abs(obj.gY))) ];
            obj.record.gU = [ obj.record.gU max(max(abs(obj.gU))) ];
            obj.record.epsilon = [ obj.record.epsilon obj.epsilon ];
            obj.record.sigma = [ obj.record.sigma obj.sigma ];
            obj.record.sparse = [ obj.record.sparse obj.sparse ];
            obj.record.loss   = [ obj.record.loss obj.loss ];
            obj.record.time   = [ obj.record.time toc ];
    
            if obj.verbose
                fprintf('gX = %g, gY = %g, gU = %g, epsilon = %g, norm(XT*X-I): %3.2e\n', ...
                max(max(abs(obj.gX))),max(max(abs(obj.gY))), max(max(abs(obj.gU))), obj.epsilon,norm(obj.X'*obj.X - eye(obj.r), 'fro'));
            end
        end

        function obj = run(obj, tol,kkt_tol)
            tic;
            for i = 1:250
                obj = obj.update();
                if obj.verbose
                    fprintf("Iter = %d, sigma = %g, gap = %g, sparse = %f , loss = %g\n, time = %f\n", i, obj.sigma, max(max(abs(obj.X - obj.Y))), obj.sparse, obj.loss, toc);
                end
                if max(max(abs(obj.gU))) < tol && max(max(abs(obj.gX))) < kkt_tol
                    break
                end
            end
            time = toc;
            obj.elapsed_time = time;
%             [fv, ind] = min(obj.record.gX);
            fprintf('%s:Iter ***  Fval *** CPU  **** sparsity *** opt_norm *** tcg_count\n', obj.algname);
            print_format = ' %i     %1.5e    %1.2f     %1.4f          %1.4e  %d \n';
%             fprintf(1,print_format, obj.iter_num, obj.record.loss(ind), time, obj.record.sparse(ind), obj.record.gX(ind),obj.tcg_count);
            fprintf(1,print_format, obj.iter_num, obj.record.loss(end), time, obj.record.sparse(end), obj.record.gX(end),obj.tcg_count);
            % for observation
            if obj.verbose
                fprintf("back_situation1_count = %d, back_situation2_count = %d, back_situation3_count = %d, RGBB_USE_COUNT = %d\n", obj.back_situation1_count, obj.back_situation2_count, obj.back_situation3_count, obj.RGBB_COUNT);
            end
        end


        function obj = init_rand(obj, H, options)
            obj.H = -H;
            obj.mu = options.mu;
            obj.n = options.n;
            obj.r = options.r;
            obj.tau = options.tau;
%             obj.Sv = svds(full(H), 1);
            obj.sigma = 1;  %(obj.Sv ^ 2) * 3;% 
            obj.maxtr_iter = options.maxtr_iter;
            obj.sigma_factor = options.sigma_factor;
            obj.manifold = stiefelfactory(obj.n, obj.r);
            obj.verbose = options.verbose;
            obj.epsilon = 1e-2;
            obj.epsilon_min = 1e-13;
            obj.delta = 1e-2;
            obj.delta_max = 10;
            obj.delta_min = -Inf;%1e-8;
            obj.rho0 = 0.1;
            obj.tcg_normrr_tol = -Inf;%7e-5;
            obj.zeros_eta = zeros(obj.n,obj.r);
            obj.eta_min = -Inf;
            obj.tcg_count = 0;
            obj.minGX = Inf;
            obj.rec_c = 0;
            obj.rec_c0 = 0;
            obj.rec_maxtr_iter = obj.maxtr_iter;
            obj.fail_count = 0;
            obj.RGBB_COUNT = 0;

            % for observe
            obj.back_situation1_count = 0;
            obj.back_situation2_count = 0;
            obj.back_situation3_count = 0;
            

%             obj.X = obj.manifold.rand();
            obj.Y = zeros([obj.n obj.r]);
            obj.U = zeros([obj.n obj.r]);
            obj.iter_num = 1;
            obj.residual = 1.0e10;
    
            if isfield(options, 'retraction')
                if options.retraction == "exp"
                    obj.retraction = obj.manifold.exp;
                elseif options.retraction == "retr"
                    obj.retraction = obj.manifold.retr;
                elseif options.retraction == "polar"
                    obj.retraction = obj.manifold.retr_polar;
                else
                    obj.retraction = obj.manifold.retr;
                end
            else
                obj.retraction = obj.manifold.retr;
            end
    
            if isfield(options, 'maxtcg_iter')
                obj.maxtcg_iter = options.maxtcg_iter;
            else
                obj.maxtcg_iter = 100;
            end
    
            if isfield(options, 'algname')
                obj.algname = options.algname;
            else
                obj.algname = 'ALMTR';
            end
    
    
            obj.record.gX = [];
            obj.record.gY = [];
            obj.record.gU = [];
            obj.record.epsilon = [];
            obj.record.sigma = [];
            obj.record.sparse = [];
            obj.record.loss = [];
            obj.record.time = [];
    
        end
        
        function obj = init(obj, A, options)
            obj = obj.init_rand(A, options);
            obj.U = zeros([obj.n obj.r]);
            obj.X = options.x_init;
            obj.Y = obj.l1_prox(obj.X + obj.U / obj.sigma, obj.mu / obj.sigma);
        end
    end
end

