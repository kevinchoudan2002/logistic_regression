% Conjugate gradiant for solving Ax=-g

function [x, r, iter] = CG(A,g,CG_tol,max_iter)
    x = 0; 
    r = g; p = -r;
    for iter = 1:max_iter
        rr = r'* r;
        Ap = A * p;
        alpha = rr / (p'*Ap);
        x = x + alpha * p;
        r = r + alpha * Ap; % r = Ax + g is residual
        nr1 = norm(r);
        if nr1 <= CG_tol
            break;
        end
        beta = nr1^2 / rr;
        p = -r + beta * p;
    end
    % fprintf('CG_total_iter_number = %d\n', iter);
end