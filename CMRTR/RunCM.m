mkdir('log');
dfile=['log/cm_', datestr(now,'yyyy-mm-dd-HHMMSS'),'.log'];
diary(dfile);
diary on;

RunRep(1000,  50, 20, 0.10, 1, 10);
% RunRep(200,  50, 20, 0.10, 1, 20);
% RunRep(500,  50, 20, 0.10, 1, 20);
% RunRep(1000, 50, 20, 0.10, 1, 20);
% RunRep(1500, 50, 20, 0.10, 1, 20);
% RunRep(2000, 50, 20, 0.10, 1, 20);
% %
% RunRep(1000, 50, 10, 0.10, 1, 20);
% RunRep(1000, 50, 15, 0.10, 1, 20);
% RunRep(1000, 50, 25, 0.10, 1, 20);
% RunRep(1000, 50, 30, 0.10, 1, 20);
% 
% RunRep(1000, 50, 20, 0.05, 1, 20);
% RunRep(1000, 50, 20, 0.15, 1, 20);
% RunRep(1000, 50, 20, 0.20, 1, 20);
% RunRep(1000, 50, 20, 0.25, 1, 20);
% 
% RunRep(500, 50, 50, 0.05, 1, 20);
% RunRep(500, 50, 35, 0.05, 1, 20);
% RunRep(500, 50, 20, 0.05, 1, 20);
%

diary off;

function RunRep(n, m, r, lambda, start, reps)

    T0 = []; 
    S0 = []; 
    F0 = [];
    P0 = [];
    ON0 = [];
    TCGNUM0 = [];



%     prefix = sprintf('data/%d-%d-%d-%.2f', n, m, r, lambda);
    

    for num = start:reps

        seed = 1 * num;
        rng(seed);
        
        fprintf(1, '\n   --------- #%d/%d, n = %d, r = %d, mu = %.2f ---------\n', num, reps, n, r, lambda);

        dx = m/n;  V = 0;
        Lu = 8/dx^2.*(sin(pi/4))^2;
        Ll = Lu*0.8;
        A = -Sch_matrix(0,m,n); %  schrodinger operator

        [phi_init,~] = svd(randn(n,r),0);  % random intialization

        Xinitial.main = phi_init;

        dx = m/n;  V = 0;

%        tol = 1e-8*n*r;
        tol = 5e-5;
        option_tr.mu = lambda;
        option_tr.n = n; option_tr.r = r;
        option_tr.tau = 0.97;
        option_tr.sigma_factor = 1.25;
        option_tr.maxtr_iter = 100;
        option_tr.retraction = 'retr';
        option_tr.verbose = 1;
        option_tr.x_init = Xinitial.main;
        option_tr.gradnorm_decay = 0.95; 
        option_tr.gradnorm_min = 1.0e-13;
        tr_solver = CMRTR();
        tr_solver = tr_solver.init(A, option_tr);

        tic;
        tr_solver = tr_solver.run(tol);
        time0 = toc;

        record0 = tr_solver.record;
        xopt0 = tr_solver.X; fv0 = record0.loss(end); sparsity0 = record0.sparse(end); 
        optgradnorm = tr_solver.record.gX(end);

        T0 = [T0; time0];
        F0 = [F0; fv0];
        S0 = [S0; sparsity0 * 100];
        ON0 = [ON0; optgradnorm];
        TCGNUM0 = [TCGNUM0; tr_solver.tcg_count];

    end

    fprintf(1, '=========== Summary: n = %d, r = %d, mu = %.3f ==========\n', n, r, lambda);
    fprintf(1, 'ALMTR:    time = %.3fs,  sparsity = %.2f,  optgradnorm = %1.3e,  loss = %.2f, tcg_count = %d \n',       mean(T0), mean(S0), mean(ON0),mean(F0),mean(TCGNUM0));
end


%    [time3, time4, time6]

