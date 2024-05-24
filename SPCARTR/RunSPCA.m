addpath ./manopt;

mkdir('log');
dfile=['log/spca_', datestr(now,'yyyy-mm-dd-HHMMSS'),'.log'];
diary(dfile);
diary on;


RunRep(500,  50, 20, 1.00, 1, 1);
% RunRep(500,  50, 20, 1.00, 1, 20);
% RunRep(1000, 50, 20, 1.00, 1, 20);
% RunRep(1500, 50, 20, 1.00, 1, 20);
% RunRep(2000, 50, 20, 1.00, 1, 20);
% RunRep(2500, 50, 20, 1.00, 1, 20);
% RunRep(3000, 50, 20, 1.00, 1, 20);
% %%
% RunRep(2000, 50, 5,  1.00, 1, 20);
% RunRep(2000, 50, 10, 1.00, 1, 20);
% RunRep(2000, 50, 15, 1.00, 1, 20);
% RunRep(2000, 50, 25, 1.00, 1, 20);
% %
% RunRep(2000, 50, 20, 0.25, 1, 20);
% RunRep(2000, 50, 20, 0.50, 1, 20);
% RunRep(2000, 50, 20, 0.75, 1, 20);
% RunRep(2000, 50, 20, 1.25, 1, 20);

diary off;

function RunRep(n, m, r, lambda, start, reps)

    T0 = []; T1 = []; 
    S0 = []; S1 = []; 
    F0 = []; F1 = []; 
    P0 = [];
    ON0 = [];
    TCGNUM0 = [];


    for num = start:reps

        fprintf(1, '\n   --------- #%d/%d, n = %d, r = %d, mu = %.2f ---------\n', num, reps, n, r, lambda);

        rng(num * 20);
        A = randn(m, n);
        A = A - repmat(mean(A, 1), m, 1);
        A = A ./ repmat(sqrt(sum(A .* A)), m, 1);
        
        [U, S, V] = svd(A, 'econ');
        D = diag(S(1:r, 1:r));
        D = sort(abs(randn([m 1])) .^ 4) + 1.0e-5;

        A = U * diag(D) * V';
        A = A - repmat(mean(A, 1), m, 1);
        A = A ./ repmat(sqrt(sum(A .* A)), m, 1);

        [phi_init,~] = svd(randn(n,r),0);  % random intialization

        Xinitial.main = phi_init;

        tol = 1e-8;
        
        option_tr.mu = lambda;
        option_tr.n = n; option_tr.r = r;
        option_tr.tau = 0.9;
        option_tr.sigma_factor = 1.25;
        option_tr.maxinner_iter = 300;
        option_tr.maxtr_iter = 60;%50;
        option_tr.retraction = 'retr';
        option_tr.verbose = 1;
        option_tr.x_init = Xinitial.main;
        tr_solver = SPCARTR();
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
