
"""
soc_proj! is a function that projects the vector onto the second order cone.
"""
function soc_proj!(sol::AbstractVector{rpdhg_float})
    t = sol[1]
    x = @view sol[2:end]
    nrm_x = LinearAlgebra.norm(x)
    if nrm_x <= -t
        sol .= 0.0
    elseif nrm_x <= t
        return
    else
        c = (1.0 + t/nrm_x)/2.0
        sol[1] = nrm_x * c
        x .= x * c
    end
end


"""
rsoc_proj! is a function that projects the vector onto the rotated second order cone.
"""
function process_lambd1(x0, y0, C)
    # solving 0.5 * (x - x0)^2 + 0.5 * (y - y0)^2 s.t. x >= 0, y >= 0, x * y = C
    if C == 0
        # case 1: x = 0, y = max(y0, 0)
        x1 = 0.0
        y1 = max(y0, 0)
        obj1 = 0.5 * (x1 - x0)^2 + 0.5 * (y1 - y0)^2
        # case 2: x = max(x0, 0), y = 0
        x2 = max(x0, 0)
        y2 = 0.0
        obj2 = 0.5 * (x2 - x0)^2 + 0.5 * (y2 - y0)^2
        if obj1 < obj2
            return x1, y1
        else
            return x2, y2
        end
    else
        # case 3 min 0.5 * (x - x0)^2 + 0.5 * (y - y0)^2 s.t. x >= 0, y >= 0, x * y = C
        # solving x^4 - x0 * x^3 + c*y0 * x - C^2 = 0
        p = Polynomial([-C^2, C*y0, 0.0, -x0, 1.0])
        polynomial_roots = roots(p)
        polynomial_roots = filter(x-> imag(x) < 1e-6, polynomial_roots)
        polynomial_roots = real.(polynomial_roots)
        polynomial_roots = filter(x -> x > 0.0, polynomial_roots)
        len_roots = length(polynomial_roots)
        if len_roots == 0
            error("No real roots found for the equation.")
        elseif len_roots == 1
            x = polynomial_roots[1]
            y = C / x
        else
            min_obj = Inf
            for root in polynomial_roots
                obj = 0.5 * (root - x0)^2 + 0.5 * (C / root - y0)^2
                if obj < min_obj
                    min_obj = obj
                    x = root
                    y = C / root
                end
            end
        end
        return x, y
    end
end

function solve_quadratic(a, b, c)
    discriminant = b^2 - 4a*c
    if discriminant < 0
        println("discriminant: ", discriminant)
        println("a: ", a, " b: ", b, " c: ", c)
        throw(ArgumentError("The discriminant is negative, so the roots are complex numbers."))
    elseif discriminant == 0
        return 1, -b / (2a), Inf
    else
        val1 = (-b + sqrt(discriminant)) / (2a)
        val2 = (-b - sqrt(discriminant)) / (2a)
        return 2, val1, val2
    end
end

function rsoc_proj!(x::AbstractVector{rpdhg_float})
    x0 = x[1]
    y0 = x[2]
    z0 = @view x[3:end]
    x0y0 = x0 * y0
    x0Squr = x0^2
    y0Squr = y0^2
    z0Nrm = norm(z0)
    z0NrmSqur = z0Nrm^2

    if (2 * x0y0 > z0NrmSqur && x0 >= 0 && y0 >= 0)
        return
    end
    if x0 <= 0 && y0 <= 0 && 2 * x0y0 >= z0NrmSqur
        x .= 0.0
        return
    end

    if (abs(x0+y0) < positive_zero)
        z0 ./= 2
        x[1], x[2] = process_lambd1(x[1], x[2], norm(z0)^2 / 2)
        return
    end
    alpha = z0NrmSqur - 2 * x0y0
    beta = -2 * (z0NrmSqur + x0Squr + y0Squr)
    rootNum, roots1, roots2 = solve_quadratic(alpha, beta, alpha)
    if rootNum == 1
        lambd = roots1
        if (abs(lambd - 1) < positive_zero)
            z0 ./= 2
            x[1], x[2] = process_lambd1(x[1], x[2], norm(z0)^2 / 2)
            return
        end
        denominator = (1 - lambd^2)
        xNew = (x0 + lambd * y0) / denominator
        yNew = (y0 + lambd * x0) / denominator
        x[1] = xNew
        x[2] = yNew
        z0 ./= denominator
        return
    elseif rootNum == 2
        lambd1 = roots1
        lambd2 = roots2
        denominator1 = (1 - lambd1^2)
        denominator2 = (1 - lambd2^2)
        xNew1 = (x0 + lambd1 * y0) / denominator1
        yNew1 = (y0 + lambd1 * x0) / denominator1
        xNew2 = (x0 + lambd2 * y0) / denominator2
        yNew2 = (y0 + lambd2 * x0) / denominator2
        z0New1 = z0 / (1 + lambd1)
        z0New2 = z0 / (1 + lambd2)
        if (xNew1 > negative_zero && yNew1 > negative_zero)
            if (xNew2 > negative_zero && yNew2 > negative_zero)
                # two point are feasible
                v1 = [xNew1; yNew1; z0New1...]
                v2 = [xNew2; yNew2; z0New2...]
                if norm(v1 - x) < norm(v2 - x)
                    x[1] = xNew1
                    x[2] = yNew1
                    z0 .= z0New1
                else
                    x[1] = xNew2
                    x[2] = yNew2
                    z0 .= z0New2
                end
                return
            else
                x[1] = xNew1
                x[2] = yNew1
                z0 .= z0New1
                return
            end
        else
            if (xNew2 > negative_zero && yNew2 > negative_zero)
                x[1] = xNew2
                x[2] = yNew2
                z0 .= z0New2
                return
            else
                x .= 0
                return
            end
        end
    else
        throw(ArgumentError("The root number is not 1 or 2."))
    end
end

# function oracle_soc_f(xi, x, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part)
#     temp_part .= 1 ./ (1 .+ ((2 * xi) .* D_scaled_squared_part))
#     temp_part .*= D_scaled_part_mul_x_part
#     left = norm(temp_part)^2
#     right = (x[1] / (1 - 2 * xi))^2
#     return left - right
# end

function oracle_soc_f_sqrt(xi, x, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part)
    temp_part .= 1 ./ (1 .+ ((2 * xi) .* D_scaled_squared_part))
    temp_part .*= D_scaled_part_mul_x_part
    left = norm(temp_part)
    right = (x[1] / (1 - 2 * xi))
    return left - right
end

function oracle_soc_h(xi, x, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part)
    temp_part .= 1 ./ (1 .+ ((2 * xi) .* D_scaled_squared_part))
    temp_part .*= D_scaled_part_mul_x_part
    left = norm(temp_part)^2
    right = (x[1] / (1 - 2 * xi))^2
    f = left - right
    temp_part ./= sqrt.(2 * xi .+ D_scaled_squared_part)
    right /= (1 - 2 * xi)
    h = -4 * (norm(temp_part)^2 + right)
    return f, h
end

function newton_soc_rootsearch(xiLeft, xiRight, xi, sol, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part)
    converged = false
    LODAMP = 0.05
    HIDAMP = 0.95
    for i = 1:20
        f, df = oracle_soc_h(xi, sol, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part)
        if f < 0
            xiRight = xi
        else
            xiLeft = xi
        end
        if (xiRight <= xiLeft)
            converged = true
            break
        end
        if (isfinite(f) && df < -proj_rel_tol)
            xi = xi - f/df
        else
            break
        end
        if (abs(f) <= proj_abs_tol)
            converged = true
            break
        end
        xi = clamp(xi, xiLeft+proj_rel_tol, xiRight-proj_rel_tol)
    end
    return converged, xi, xiLeft, xiRight
end

function soc_proj_diagonal!(sol::AbstractVector{rpdhg_float},
    x2end::AbstractVector{rpdhg_float},
    D_scaled_part::AbstractVector{rpdhg_float}, 
    D_scaled_squared_part::AbstractVector{rpdhg_float}, 
    D_scaled_part_mul_x_part::AbstractVector{rpdhg_float}, 
    temp_part::AbstractVector{rpdhg_float},
    t_warm_start::AbstractVector{rpdhg_float}, i::Integer, 
    projInfo::timesInfo)
    """
    solve the projection problem for the second order cone
        min 0.5 * ||x - y||^2 + 0.5 * (t - s)^2
        s.t. ||D^{-1} y||<= d^{-1} * s <- xi is dual variable
            iff ||d D^{-1} y||^2 <= s^2
            iff (D^{-1} y) in second order cone
    # D_scaled = D[1] ./ D
    """
    # println("x: ", sol, "D_scaled_part: ", D_scaled_part)
    # xCopy = deepcopy(sol)
    # minVal is the minimum value of the absolute values of the elements in sol but not zero
    minVal = filter(x -> x > 0, min.(abs.(sol)))
    if length(minVal) == 0
        minVal = 1.0
    else
        minVal = max(minimum(minVal), 1e-3)
    end
    sol ./= minVal
    t = sol[1]
    temp_part .= x2end./D_scaled_part 
    if norm(temp_part) <= -sol[1] && sol[1] <= 0
        x2end .= 0.0
        sol[1] = 0.0
        projInfo.zero += 1
        projInfo.status = :zero
        return
    end
    D_scaled_part_mul_x_part .= D_scaled_part .* x2end
    if norm(D_scaled_part_mul_x_part) <= sol[1]
        sol[1] = max(sol[1], 0)
        sol .*= minVal
        # if isinf(sol[1]) || isnan(sol[1])
        #     println("norm(D_scaled_part_mul_x_part) <= sol[1] small sol[1] is inf or nan: ", sol[1])
        #     error("norm(D_scaled_part_mul_x_part) <= sol[1] small sol[1] is inf or nan")
        # end
        projInfo.interior += 1
        projInfo.status = :interior
        return
    end
    if t > proj_rel_tol
        xiRight = 0.5
        xiLeft = 0.0
        oracleVal = 1.0
        if t_warm_start[i] > xiLeft && t_warm_start[i] < xiRight
            oracleVal = oracle_soc_f_sqrt(t_warm_start[i], sol, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part)
            if abs(oracleVal) < proj_abs_tol
                xiMid = t_warm_start[i]
                sol[1] = sol[1] / (1 - 2 * xiMid)
                x2end ./= (1 .+ 2 .* xiMid .* D_scaled_squared_part)
                sol .*= minVal
                # if isinf(sol[1]) || isnan(sol[1])
                #     println("abs small sol[1] is inf or nan: ", sol[1])
                #     error("abs small sol[1] is inf or nan")
                # end
                projInfo.boundary += 1
                projInfo.status = :boundary
                return
            end
            if oracleVal < 0
                xiRight = t_warm_start[i]
            else
                xiLeft = t_warm_start[i]
            end
        end
        xi = (xiRight + xiLeft) / 2
        converged, xiMid, xiLeft, xiRight = newton_soc_rootsearch(xiLeft, xiRight, xi, sol, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part)
        # println("converged: ", converged, " xiMid: ", xiMid)
        converged = false
        if !converged
            while (xiRight - xiLeft) / (1 + xiRight + xiLeft) > proj_rel_tol && abs(oracleVal) > proj_abs_tol
                xiMid = (xiRight + xiLeft) / 2
                oracleVal = oracle_soc_f_sqrt(xiMid, sol, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part)
                if oracleVal < 0
                    xiRight = xiMid
                else
                    xiLeft = xiMid
                end
            end
        end
        t_warm_start[i] = xiMid
        sol[1] = sol[1] / (1 - 2 * xiMid)
        x2end ./= (1 .+ 2 .* xiMid .* D_scaled_squared_part)
        sol .*= minVal
        # if isinf(sol[1]) || isnan(sol[1])
        #     println("t > proj_rel_tol sol[1] is inf or nan: ", sol[1])
        #     error("t > proj_rel_tol sol[1] is inf or nan")
        # end
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    elseif t < -proj_rel_tol
        xiRight = 1
        xiLeft = 0.5
        while oracle_soc_f_sqrt(xiRight, sol, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part) < 0
            xiLeft = xiRight
            xiRight *= 2
        end
        oracleVal = 1.0
        while (xiRight - xiLeft) / (1 + xiRight + xiLeft) > proj_rel_tol && abs(oracleVal) > proj_abs_tol
            xiMid = (xiRight + xiLeft) / 2
            oracleVal = oracle_soc_f_sqrt(xiMid, sol, D_scaled_part_mul_x_part, D_scaled_squared_part, temp_part)
            # println("t <= 0 oracleVal: ", oracleVal, " xiMid: ", xiMid)
            if oracleVal < 0
                xiLeft = xiMid
            else
                xiRight = xiMid
            end
        end
        sol[1] = sol[1] / (1 - 2 * xiMid)
        x2end ./= (1 .+ 2 .* xiMid .* D_scaled_squared_part)
        sol .*= minVal
        # if isinf(sol[1]) || isnan(sol[1])
        #     println("xCopy: ", xCopy)
        #     println("D_scaled_part: ", D_scaled_part)
        #     println("final sol[1] is inf or nan: ", sol[1])
        #     error("final sol[1] is inf or nan")
        # end
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    else
        # t == 0
        x2end ./= (1 .+ D_scaled_squared_part)
        temp_part .= D_scaled_part .* x2end
        sol[1] = norm(temp_part)
        sol .*= minVal
        # if isinf(sol[1]) || isnan(sol[1])
        #     println("xCopy: ", xCopy)
        #     println("D_scaled_part: ", D_scaled_part)
        #     println("final sol[1] is inf or nan: ", sol[1])
        #     error("final sol[1] is inf or nan")
        # end
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    end # end if
end # end soc_proj_diagonal!



function soc_proj_const_scale!(sol::AbstractVector{rpdhg_float},
    x2end::AbstractVector{rpdhg_float},
    D_scaled_part::AbstractVector{rpdhg_float}, 
    D_scaled_squared_part::AbstractVector{rpdhg_float}, 
    D_scaled_part_mul_x_part::AbstractVector{rpdhg_float}, 
    temp_part::AbstractVector{rpdhg_float},
    t_warm_start::AbstractVector{rpdhg_float}, i::Integer,
    projInfo::timesInfo)
    t = sol[1]
    nrm_x = LinearAlgebra.norm(x2end)
    if nrm_x <= -t
        sol .= 0.0
        projInfo.zero += 1
        projInfo.status = :zero
        return
    elseif nrm_x <= t
        projInfo.interior += 1
        projInfo.status = :interior
        return
    else
        c = (1.0 + t/nrm_x)/2.0
        sol[1] = nrm_x * c
        x2end .= x2end * c
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    end
end

# function oracle_rsoc_f(xi, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
#     temp_part .= x_mul_d_part ./ (1 .+ xi .* D_scaled_squared_part)
#     left = norm(temp_part)^2
#     xi_sqr = xi^2
#     xi_sqr_one = xi_sqr - 1
#     xi_sqr_one_sqr = xi_sqr_one^2
#     right = 2 * (x0y0 + (x0_sqr + y0_sqr) * xi + x0y0 * xi_sqr) / xi_sqr_one_sqr
#     f = left - right
#     return f
# end

function oracle_rsoc_f_sqrt(xi, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
    temp_part .= x_mul_d_part ./ (1 .+ xi .* D_scaled_squared_part)
    left = norm(temp_part)
    xi_sqr = xi^2
    xi_sqr_one = xi_sqr - 1
    xi_sqr_one_sqr = xi_sqr_one^2
    right = 2 * (x0y0 + (x0_sqr + y0_sqr) * xi + x0y0 * xi_sqr) / xi_sqr_one_sqr
    f = left - sqrt(right)
    return f
end

function oracle_rsoc_h(xi, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_part, D_scaled_squared_part, temp_part)
    temp_part .= x_mul_d_part ./ (1 .+ xi .* D_scaled_squared_part)
    left = norm(temp_part)^2
    xi_sqr = xi^2
    xi_sqr_one = xi_sqr - 1
    xi_sqr_one_sqr = xi_sqr_one^2
    right = 2 * (x0y0 + (x0_sqr + y0_sqr) * xi + x0y0 * xi_sqr) / xi_sqr_one_sqr
    f = left - right

    temp_part ./= sqrt.(1 .+ xi .* D_scaled_squared_part)
    temp_part .*= D_scaled_part
    h_left = -2 * norm(temp_part)^2
    h_right1 = 2 * (2 * x0y0 * xi + x0_sqr + y0_sqr) / xi_sqr_one_sqr
    h_right2 = 8 * (x0y0 + (x0_sqr + y0_sqr) * xi + x0y0 * xi_sqr) * xi_sqr_one * xi / xi_sqr_one_sqr^2
    h = h_left - h_right1 + h_right2
    return f, h
end

function newton_rsoc_rootsearch(xiLeft, xiRight, xi, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_part, D_scaled_squared_part, temp_part)
    converged = false
    LODAMP = 0.05
    HIDAMP = 0.95
    for i = 1:20
        f, df = oracle_rsoc_h(xi, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_part, D_scaled_squared_part, temp_part)
        # println("f: ", f, " df: ", df, " xi: ", xi)
        if f < 0
            xiRight = xi
        else
            xiLeft = xi
        end
        if (xiRight <= xiLeft)
            converged = true
            break
        end
        if (isfinite(f) && df < -proj_rel_tol)
            xi = xi - f/df
        else
            break
        end
        if (abs(f) <= proj_abs_tol)
            converged = true
            break
        end
        xi = clamp(xi, xiLeft+proj_rel_tol, xiRight-proj_rel_tol)
    end
    return converged, xi, xiLeft, xiRight
end


function rsoc_proj_diagonal!(x::AbstractVector{rpdhg_float},
    sol_part::AbstractVector{rpdhg_float},
    D_scaled::AbstractVector{rpdhg_float},
    D_scaled_part::AbstractVector{rpdhg_float},
    D_scaled_squared_part::AbstractVector{rpdhg_float},
    x_mul_d_part::AbstractVector{rpdhg_float},
    temp_part::AbstractVector{rpdhg_float},
    t_warm_start::AbstractVector{rpdhg_float}, i::Integer,
    projInfo::timesInfo)
    """
    solve the projection problem for the second order cone
        min ||z - z0||^2 + (x - x0)^2 + (y - y0)^2 
        s.t. ||D^{-1} z||^2<= 2 d^{-1}_x * d^{-1}_y x y<- xi is dual variable
                x >0 , y > 0
            iff d_x d_y || D^{-1} z||^2 <= 2 x y, x > 0, y > 0
    # D_scaled = sqrt{d_x d_y} ./ D
    """
    minVal = filter(x -> x > 0, min.(abs.(x)))
    if length(minVal) == 0
        minVal = 1.0
    else
        minVal = max(minimum(minVal), 1e-3)
    end
    x ./= minVal
    x0 = x[1]
    y0 = x[2]

    dx0 = D_scaled[1]
    dy0 = D_scaled[2]
    x_mul_d_part .= sol_part .* D_scaled_part
    temp_part .= D_scaled_part .* sol_part
    z0NrmSqur = norm(temp_part)^2
    if (2 * x0 * y0 >= z0NrmSqur && x0 >= 0 && y0 >= 0)
        x .*= minVal
        projInfo.interior += 1
        projInfo.status = :interior
        return
    end

    temp_part .= sol_part ./ D_scaled_part
    val = norm(temp_part)^2
    if (x0 <= 0 && y0 <= 0 && 2 * x0 * y0 > val)
        x .= 0.0
        projInfo.zero += 1
        projInfo.status = :zero
        return
    end

    if (abs(x0+y0) < positive_zero)
        sol_part ./= (1 .+ D_scaled_squared_part)
        temp_part .= sol_part .* D_scaled_part
        val_temp = norm(temp_part)^2 / 2
        x[1], x[2] = process_lambd1(x[1], x[2], val_temp)
        x .*= minVal
        # if isinf(x[1]) || isinf(x[2]) || isnan(x[1]) || isnan(x[2])
        #     println("abs(x0+y0) < positive_zero x: ", x)
        # end
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    end

    x0_sqr = x0^2
    y0_sqr = y0^2
    x0y0 = x0 * y0

    if x0 > 0 && y0 > 0
        xiRight = 1.0
        xiLeft = 0
        oracle_fVal = 1.0
        xiMid = (xiRight + xiLeft) / 2
        if t_warm_start[i] > xiLeft && t_warm_start[i] < xiRight
            oracle_fVal = oracle_rsoc_f_sqrt(t_warm_start[i], x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
            if abs(oracle_fVal) < proj_abs_tol
                xi = t_warm_start[i]
                x[1] = (x0 + y0 * xi) / (1 - xi^2)
                x[2] = (y0 + x0 * xi) / (1 - xi^2)
                sol_part ./= (1 .+ xi .* D_scaled_squared_part)
                x .*= minVal
                # if isinf(x[1]) || isinf(x[2]) || isnan(x[1]) || isnan(x[2])
                #     println("x0 > 0 && y0 > 0 x: ", x)
                # end
                projInfo.boundary += 1
                projInfo.status = :boundary
                return
            end
            if oracle_fVal < 0
                xiRight = t_warm_start[i]
            else
                xiLeft = t_warm_start[i]
            end
        end
        xi = (xiRight + xiLeft) / 2
        converged, xiMid, xiLeft, xiRight = newton_rsoc_rootsearch(xiLeft, xiRight, xi, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_part, D_scaled_squared_part, temp_part)
        # println("converged: ", converged, " xiMid: ", xiMid, " xiLeft: ", xiLeft, " xiRight: ", xiRight)
        converged = false
        if !converged
            while (xiRight - xiLeft) / (1 + xiRight + xiLeft) > 1e-14 && abs(oracle_fVal) > proj_rel_tol
                xiMid = (xiRight + xiLeft) / 2
                oracle_fVal = oracle_rsoc_f_sqrt(xiMid, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
                # println("x0 > 0 && y0 > 0 oracle_fVal: ", oracle_fVal, " xiMid: ", xiMid)
                if oracle_fVal < 0
                    xiRight = xiMid
                else
                    xiLeft = xiMid
                end
            end
        end
        xi = xiMid
        t_warm_start[i] = xi
        x[1] = (x0 + y0 * xi) / (1 - xi^2)
        x[2] = (y0 + x0 * xi) / (1 - xi^2)
        sol_part ./= (1 .+ xi .* D_scaled_squared_part)
        x .*= minVal
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    elseif x0 < 0 && y0 < 0
        temp_part .= sol_part ./ D_scaled_part
        val = norm(temp_part)^2
        if 2 * x0 * y0 > val
            x .= 0.0
            return
        end
        xiRight = 2.0
        xiLeft = 1.0
        if t_warm_start[i] > xiLeft
            xiRight = t_warm_start[i]
        end
        while oracle_rsoc_f_sqrt(xiRight, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part) < 0
            xiLeft = xiRight
            xiRight *= 2
        end
        oracle_fVal = 1.0
        xiMid = (xiRight + xiLeft) / 2
        while (xiRight - xiLeft) / (1 + xiRight + xiLeft) > 1e-14 && abs(oracle_fVal) > proj_rel_tol
            xiMid = (xiRight + xiLeft) / 2
            oracle_fVal = oracle_rsoc_f_sqrt(xiMid, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
            # println("x0 < 0 && y0 <0 oracle_fVal: ", oracle_fVal, " xiMid: ", xiMid, "xiRight: ", xiRight, " xiLeft: ", xiLeft)
            if oracle_fVal > 0
                xiRight = xiMid
            else
                xiLeft = xiMid
            end
        end
        xi = xiMid
        t_warm_start[i] = xi
        x[1] = (x0 + y0 * xi) / (1 - xi^2)
        x[2] = (y0 + x0 * xi) / (1 - xi^2)
        sol_part ./= (1 .+ xi .* D_scaled_squared_part)
        x .*= minVal
        # if isinf(x[1]) || isinf(x[2]) || isnan(x[1]) || isnan(x[2])
        #     println("x0 < 0 && y0 < 0 x: ", x)
        # end
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    else
        if x0 <= 0 && y0 >= 0 && x0 + y0 <= 0
            xiRight = -x0 / y0
            xiLeft = 1.0
            if y0 == 0
                xiRight = 1.0
                if t_warm_start[i] > xiLeft
                    xiRight = t_warm_start[i]
                end
                while oracle_rsoc_f_sqrt(xiRight, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part) < 0
                    xiLeft = xiRight
                    xiRight *= 2
                end
            end
            oracle_fVal = 1.0
            xiMid = (xiRight + xiLeft) / 2
            while (xiRight - xiLeft) / (1 + xiRight + xiLeft) > 1e-14 && abs(oracle_fVal) > proj_abs_tol
                xiMid = (xiRight + xiLeft) / 2
                oracle_fVal = oracle_rsoc_f_sqrt(xiMid, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
                # println("x0 < 0 && y0 > 0 && x0 + y0 < 0 oracle_fVal: ", oracle_fVal, " xiMid: ", xiMid)
                if oracle_fVal > 0
                    xiRight = xiMid
                else
                    xiLeft = xiMid
                end
            end
        elseif x0 <= 0 && y0 >= 0 && x0 + y0 >= 0
            xiRight = 1.0
            xiLeft = -x0 / y0
            oracle_fVal = 1.0
            if t_warm_start[i] > xiLeft && t_warm_start[i] < xiRight
                oracle_fVal = oracle_rsoc_f_sqrt(t_warm_start[i], x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
                if abs(oracle_fVal) < proj_rel_tol
                    xi = t_warm_start[i]
                    x[1] = (x0 + y0 * xi) / (1 - xi^2)
                    x[2] = (y0 + x0 * xi) / (1 - xi^2)
                    sol_part ./= (1 .+ xi .* D_scaled_squared_part)
                    x .*= minVal
                    # if isinf(x[1]) || isinf(x[2]) || isnan(x[1]) || isnan(x[2])
                    #     println("x0 <= 0 && y0 >= 0 && x0 + y0 >= 0 x: ", x)
                    # end
                    projInfo.boundary += 1
                    projInfo.status = :boundary
                    return
                end
                if oracle_fVal < 0
                    xiRight = t_warm_start[i]
                else
                    xiLeft = t_warm_start[i]
                end
            end
            xiMid = (xiRight + xiLeft) / 2
            while (xiRight - xiLeft) / (1 + xiRight + xiLeft) > 1e-14 && abs(oracle_fVal) > proj_abs_tol
                xiMid = (xiRight + xiLeft) / 2
                oracle_fVal = oracle_rsoc_f_sqrt(xiMid, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
                # println("x0 < 0 && y0 > 0 && x0 + y0 > 0 oracle_fVal: ", oracle_fVal, " xiMid: ", xiMid)
                if oracle_fVal < 0
                    xiRight = xiMid
                else
                    xiLeft = xiMid
                end
            end
        elseif x0 >= 0 && y0 <= 0 && x0 + y0 <= 0
            xiLeft = 1.0
            xiRight = -y0 / x0
            if x0 == 0
                xiRight = 1.0
                if t_warm_start[i] > xiLeft && t_warm_start[i] < xiRight
                    oracle_fVal = oracle_rsoc_f_sqrt(t_warm_start[i], x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
                    if abs(oracle_fVal) < proj_rel_tol
                        xi = t_warm_start[i]
                        x[1] = (x0 + y0 * xi) / (1 - xi^2)
                        x[2] = (y0 + x0 * xi) / (1 - xi^2)
                        sol_part ./= (1 .+ xi .* D_scaled_squared_part)
                        x .*= minVal
                        # if isinf(x[1]) || isinf(x[2]) || isnan(x[1]) || isnan(x[2])
                        #     println("x0 == 0 && y0 < 0 && x0 + y0 <= 0 x: ", x)
                        # end
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                    if oracle_fVal > 0
                        xiRight = t_warm_start[i]
                    else
                        xiLeft = t_warm_start[i]
                    end
                end
                while oracle_rsoc_f_sqrt(xiRight, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part) < 0
                    xiLeft = xiRight
                    xiRight *= 2
                end
            end
            oracle_fVal = 1.0
            xiMid = (xiRight + xiLeft) / 2
            while (xiRight - xiLeft) / (1 + xiRight + xiLeft) > 1e-14 && abs(oracle_fVal) > proj_abs_tol
                xiMid = (xiRight + xiLeft) / 2
                oracle_fVal = oracle_rsoc_f_sqrt(xiMid, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
                # println("x0 > 0 && y0 < 0 && x0 + y0 < 0 oracle_fVal: ", oracle_fVal, " xiMid: ", xiMid)
                if oracle_fVal > 0
                    xiRight = xiMid
                else
                    xiLeft = xiMid
                end
            end
        elseif x0 >= 0 && y0 <= 0 && x0 + y0 >= 0
            xiRight = 1.0
            xiLeft = -y0 / x0
            oracle_fVal = 1.0
            if t_warm_start[i] > xiLeft && t_warm_start[i] < xiRight
                oracle_fVal = oracle_rsoc_f_sqrt(t_warm_start[i], x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
                if abs(oracle_fVal) < proj_abs_tol
                    xi = t_warm_start[i]
                    x[1] = (x0 + y0 * xi) / (1 - xi^2)
                    x[2] = (y0 + x0 * xi) / (1 - xi^2)
                    sol_part ./= (1 .+ xi .* D_scaled_squared_part)
                    x .*= minVal
                    # if isinf(x[1]) || isinf(x[2]) || isnan(x[1]) || isnan(x[2])
                    #     println("x0 > 0 && y0 < 0 && x0 + y0 >= 0 x: ", x)
                    # end
                    projInfo.boundary += 1
                    projInfo.status = :boundary
                    return
                end
                if oracle_fVal < 0
                    xiRight = t_warm_start[i]
                else
                    xiLeft = t_warm_start[i]
                end
            end
            xiMid = (xiRight + xiLeft) / 2
            while (xiRight - xiLeft) / (1 + xiRight + xiLeft) > proj_rel_tol && abs(oracle_fVal) > proj_abs_tol
                xiMid = (xiRight + xiLeft) / 2
                oracle_fVal = oracle_rsoc_f_sqrt(xiMid, x0_sqr, y0_sqr, x0y0, x_mul_d_part, D_scaled_squared_part, temp_part)
                # println("x0 > 0 && y0 < 0 && x0 + y0 > 0 oracle_fVal: ", oracle_fVal, " xiMid: ", xiMid)
                if oracle_fVal < 0
                    xiRight = xiMid
                else
                    xiLeft = xiMid
                end
            end
        end
        t_warm_start[i] = xiMid
        xi = xiMid
        x[1] = (x0 + y0 * xi) / (1 - xi^2)
        x[2] = (y0 + x0 * xi) / (1 - xi^2)
        sol_part ./= (1 .+ xi .* D_scaled_squared_part)
        x .*= minVal
        # if isinf(x[1]) || isinf(x[2]) || isnan(x[1]) || isnan(x[2])
        #     println("x0 >= 0 && y0 <= 0 && x0 + y0 >= 0 final x: ", x)
        # end
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    end
end




function two_choose_one_rsoc_proj!(z0New1::AbstractVector{rpdhg_float},
    z0New2::AbstractVector{rpdhg_float},
    xNew1::rpdhg_float,
    yNew1::rpdhg_float,
    xNew2::rpdhg_float,
    yNew2::rpdhg_float,
    lambd1::rpdhg_float,
    lambd2::rpdhg_float,
    x::AbstractVector{rpdhg_float},
    z0::AbstractVector{rpdhg_float})
    println("154 lambd1: ", lambd1, " lambd2: ", lambd2)
    z0New1 .= z0 ./ (1 .+ lambd1)
    z0New1 .-= z0
    dist1 = (xNew1 - x[1])^2 + (yNew1 - x[2])^2 + norm(z0New1)^2
    z0New2 .= z0 ./ (1 .+ lambd2)
    z0New2 .-= z0
    dist2 = (xNew2 - x[1])^2 + (yNew2 - x[2])^2 + norm(z0New2)^2
    if dist1 < dist2
        x[1] = xNew1
        x[2] = yNew1
        z0 .+= z0New1
    else
        x[1] = xNew2
        x[2] = yNew2
        z0 .+= z0New2
    end
end


function rsoc_proj_const_scale!(x::AbstractVector{rpdhg_float},
    sol_part::AbstractVector{rpdhg_float},
    D_scaled::AbstractVector{rpdhg_float},
    D_scaled_part::AbstractVector{rpdhg_float},
    D_scaled_squared_part::AbstractVector{rpdhg_float},
    x_mul_d_part::AbstractVector{rpdhg_float},
    temp_part::AbstractVector{rpdhg_float},
    t_warm_start::AbstractVector{rpdhg_float}, i::Integer,
    projInfo::timesInfo)
    minVal = filter(x -> x > 0, min.(abs.(x)))
    if length(minVal) == 0
        minVal = 1.0
    else
        minVal = max(minimum(minVal), 1e-3)
    end
    x ./= minVal
    x0 = x[1]
    y0 = x[2]
    z0 = sol_part
    x0y0 = x0 * y0
    x0Squr = x0^2
    y0Squr = y0^2
    z0Nrm = norm(z0)
    z0NrmSqur = z0Nrm^2

    z0New1 = x_mul_d_part
    z0New2 = temp_part

    if (2 * x0y0 > z0NrmSqur && x0 >= 0 && y0 >= 0)
        # println("2 * x0y0 > z0NrmSqur && x0 >= 0 && y0 >= 0")
        x .*= minVal
        projInfo.interior += 1
        projInfo.status = :interior
        return
    end
    if x0 <= 0 && y0 <= 0 && 2 * x0y0 >= z0NrmSqur
        x .= 0.0
        projInfo.zero += 1
        projInfo.status = :zero
        return
    end

    if (abs(x0+y0) < positive_zero)
        z0 ./= 2
        x[1], x[2] = process_lambd1(x[1], x[2], norm(z0)^2 / 2)
        # println("abs(x0+y0) < positive_zero")
        x .*= minVal
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    end
    alpha = z0NrmSqur - 2 * x0y0
    beta = -2 * (z0NrmSqur + x0Squr + y0Squr)
    rootNum, roots1, roots2 = solve_quadratic(alpha, beta, alpha)
    if rootNum == 1
        lambd = roots1
        if (abs(lambd - 1) < positive_zero)
            z0 ./= 2
            x[1], x[2] = process_lambd1(x[1], x[2], norm(z0)^2 / 2)
            # println("abs(lambd - 1) < positive_zero")
            x .*= minVal
            projInfo.boundary += 1
            projInfo.status = :boundary
            return
        end
        denominator = (1 - lambd^2)
        xNew = (x0 + lambd * y0) / denominator
        yNew = (y0 + lambd * x0) / denominator
        x[1] = xNew
        x[2] = yNew 
        z0 ./= denominator
        x .*= minVal
        projInfo.boundary += 1
        projInfo.status = :boundary
        return
    elseif rootNum == 2
        lambd1 = roots1
        lambd2 = roots2
        denominator1 = (1 - lambd1^2)
        denominator2 = (1 - lambd2^2)
        xNew1 = (x0 + lambd1 * y0) / denominator1
        yNew1 = (y0 + lambd1 * x0) / denominator1
        xNew2 = (x0 + lambd2 * y0) / denominator2
        yNew2 = (y0 + lambd2 * x0) / denominator2
        if (xNew1 > negative_zero && yNew1 > negative_zero)
            if (xNew2 > negative_zero && yNew2 > negative_zero)
                lambd1 = max(lambd1, 0)
                lambd2 = max(lambd2, 0)
                if x0 > 0 && y0 > 0
                    if lambd1 > 0 && lambd1 < 1 && lambd2 > 0 && lambd2 < 1
                        two_choose_one_rsoc_proj!(z0New1, z0New2, xNew1, yNew1, xNew2, yNew2, lambd1, lambd2, x, z0)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                    if lambd1 > 0 && lambd1 < 1
                        x[1] = xNew1
                        x[2] = yNew1
                        z0 ./= (1 + lambd1)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                    if lambd2 > 0 && lambd2 < 1
                        x[1] = xNew2
                        x[2] = yNew2
                        z0 ./= (1 + lambd2)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                end
                if x0 < 0 && y0 < 0
                    if lambd1 > 1 && lambd2 > 1
                        two_choose_one_rsoc_proj!(z0New1, z0New2, xNew1, yNew1, xNew2, yNew2, lambd1, lambd2, x, z0)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                    if lambd1 > 1
                        x[1] = xNew1
                        x[2] = yNew1
                        z0 ./= (1 + lambd1)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                    if lambd2 > 1
                        x[1] = xNew2
                        x[2] = yNew2
                        z0 ./= (1 + lambd2)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                end
                if (x0 < 0 && y0 > 0 && x0 + y0 < 0) || (x0 > 0 && y0 < 0 && x0 + y0 < 0)
                    upper1 = -x0 / y0
                    upper2 = -y0 / x0
                    if x0 < 0
                        upper = upper1
                    else
                        upper = upper2
                    end
                    if lambd1 > 1 && lambd1 < upper && lambd2 > 1 && lambd2 < upper
                        two_choose_one_rsoc_proj!(z0New1, z0New2, xNew1, yNew1, xNew2, yNew2, lambd1, lambd2, x, z0)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                    if lambd1 > 1 && lambd1 < upper
                        x[1] = xNew1
                        x[2] = yNew1
                        z0 ./= (1 + lambd1)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                    if lambd2 > 1 && lambd2 < upper
                        x[1] = xNew2
                        x[2] = yNew2
                        z0 ./= (1 + lambd2)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                end
                if (x0 > 0 && y0 < 0 && x0 + y0 > 0) || (x0 < 0 && y0 > 0 && x0 + y0 > 0)
                    lower1 = -x0 / y0
                    lower2 = -y0 / x0
                    if x0 < 0
                        lower = lower1
                    else
                        lower = lower2
                    end
                    if lambd1 > lower && lambd1 < 1 && lambd2 > lower && lambd2 < 1
                        two_choose_one_rsoc_proj!(z0New1, z0New2, xNew1, yNew1, xNew2, yNew2, lambd1, lambd2, x, z0)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                    if lambd1 > lower && lambd1 < 1
                        x[1] = xNew1
                        x[2] = yNew1
                        z0 ./= (1 + lambd1)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                    if lambd2 > lower && lambd2 < 1
                        x[1] = xNew2
                        x[2] = yNew2
                        z0 ./= (1 + lambd2)
                        x .*= minVal
                        projInfo.boundary += 1
                        projInfo.status = :boundary
                        return
                    end
                end
            else
                x[1] = xNew1
                x[2] = yNew1
                z0 ./= (1 + lambd1)
                x .*= minVal
                projInfo.boundary += 1
                projInfo.status = :boundary
                return
            end
        else
            if (xNew2 > negative_zero && yNew2 > negative_zero)
                x[1] = xNew2
                x[2] = yNew2
                z0 ./= (1 + lambd2)
                x .*= minVal
                projInfo.boundary += 1
                projInfo.status = :boundary
                return
            else
                x .= 0
                projInfo.zero += 1
                projInfo.status = :zero
                return
            end
        end
    else
        throw(ArgumentError("The root number is not 1 or 2."))
    end
end