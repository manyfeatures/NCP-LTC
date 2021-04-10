module wirings
export FullyConnected, add_synapse, LTCCell, LTC

using Flux
using Flux: @functor, trainable
using StatsBase # for random choice
import Random
Random.seed!(1234)

macro def(name, definition) # definition includes all the function?
  return quote
      macro $(esc(name))()
          esc($(Expr(:quote, definition)))
      end
  end
end

# Wiring
# Do we really need the exact code as for the biology?
abstract type Wiring end
@def wiring_fields begin
     units
     adjacency_matrix
     sensory_adjacency_matrix
     input_dim
     output_dim
end


# Methods
function _build(wiring::Wiring, input_shape) # for Wiring type
    _input_dim = convert(Int, input_shape[2])
    # do assert check
    @show wiring.input_dim
    if wiring.input_dim === nothing
        set_input_dim(wiring, _input_dim)
    end
    @show wiring.input_dim
end

function add_synapse(adjacency_matrix, src, dest, polarity)
    adjacency_matrix[src, dest] = polarity
    #adjacency_matrix
end

function _init_add_synapse(units, adjacency_matrix, self_conn)
    for src in (1:units)
        for dest in (1:units)
            if src === dest && !self_conn
                continue
            end
            polarity = StatsBase.sample([-1,1,1])
            add_synapse(adjacency_matrix, src, dest, polarity)
            #println(src, dest)
        end
    end
    #@show adjacency_matrix
end

function add_sensory_synapse(sensory_adjacency_matrix, src, dest, polarity)
    sensory_adjacency_matrix[src, dest] = polarity
    #adjacency_matrix
end

# not used?
function set_input_dim(wiring::Wiring, _input_dim)
    wiring.input_dim = _input_dim
    units = wiring.units
    wiring.sensory_adjacency_matrix = zeros((_input_dim, units))
end

function _erev_initializer(wiring::Wiring, shape=nothing, dtype=nothing) # dtype?
    copy(wiring.adjacency_matrix)
end

function _sensory_erev_initializer(wiring::Wiring, shape=nothing, dtype=nothing) # dtype?
    copy(wiring.sensory_adjacency_matrix)
end

function _state_size(wiring::Wiring)
    return wiring.units
end
# Test
#set_input_dim(wiring, 2)


# mutable means that it can be changed
# @with_kw allows to use default parameters
mutable struct FullyConnected <: Wiring
    @wiring_fields
    self_conns # default value ?
    #rng_ like in python
    # Inner constructor
    function FullyConnected(units, _input_dim = nothing, _output_dim = nothing, self_conns = true) #arguments order and call?
        adjacency_matrix = zeros((units, units))
        sensory_adjacency_matrix = nothing
        if _output_dim === nothing
            output_dim = units
        else
            output_dim = _output_dim
        end
        #self._rng = np.random.default_rng(erev_init_seed)
        _init_add_synapse(units, adjacency_matrix, self_conns)
        #return new{typeof{w}, typeof{uniys}, typeof{output_dim}, typeof{self_conns}}(w, units, output_dim, self_conns)
        return new(units, adjacency_matrix, sensory_adjacency_matrix, _input_dim,
                   output_dim, self_conns)
    end
end


# Methods
function build(wiring::FullyConnected, input_shape)
    _build(wiring, input_shape) # from Wiring
    @show wiring
    input_dim = wiring.input_dim
    units = wiring.input_dim
    sensory_adjacency_matrix = wiring.sensory_adjacency_matrix
    for src in (1:input_dim)
        for dest in (1:units)
#             if src === dest && !self_conn
#                 continue
#             end
            polarity = StatsBase.sample([-1,1,1])
            @show polarity
            add_sensory_synapse(sensory_adjacency_matrix, src, dest, polarity)
            #println(src, dest)
        end
    end
end


struct LTCCell
    _wiring::Wiring
    _init_ranges
    _input_mapping::String
    _output_mapping::String
    _ode_unfolds::Int
    _epsilon
    _params::Dict # Is this trainable in Dict format?
    state0 # for Recur
    function LTCCell(wiring, in_features = nothing, input_mapping="affine",
            output_mapping="affine", ode_unfolds=6, epsilon=1e-8, params=Dict(),
            state0=zeros(1, 8))
        if in_features !== nothing
            build(wiring, (nothing, in_features))
        end
        # Is this proper place?
        init_ranges = _get_init_ranges()
        params = _allocate_parameters(wiring, params, init_ranges, input_mapping, output_mapping)
        # How to launch it inside?
        new(wiring, init_ranges, input_mapping, output_mapping, ode_unfolds,
            epsilon, params, state0) # state0:(batch, units)?
        #
    end
end

#LTC(a...; ka...) = Recur(LTCCell(a...; ka...))
#Recur(m::LTCCell) = Recur(m, m.state0)

Flux.@functor LTCCell
Flux.trainable(ltc::LTCCell) = (ltc._params["sensory_w"],) # test one weight


function _get_init_ranges()
    _init_ranges = Dict(
            "gleak"=> (0.001, 1.0),
            "vleak"=> (-0.2, 0.2),
            "cm"=> (0.4, 0.6),
            "w"=> (0.001, 1.0),
            "sigma"=> (3, 8),
            "mu"=> (0.3, 0.8),
            "sensory_w"=> (0.001, 1.0),
            "sensory_sigma"=> (3, 8),
            "sensory_mu"=> (0.3, 0.8))
end


# For initializing fields
function _get_init_value(shape, name, init_ranges)
    minval, maxval = init_ranges[name]
    if minval === maxval
        return ones(shape) * minval
    else
        return rand(Float64, shape) * (maxval - minval) .+ minval
    end
end

# Not all values here need init_ranges! Decompose it
function _init_weights_and_params(wiring, name::String, init_ranges)
    state_size = _state_size(wiring)
    _sensory_size = wiring.input_dim
    if name in ["gleak", "vleak", "cm"]
        return _get_init_value(state_size, name, init_ranges)
    elseif name in ["sigma", "mu", "w"]
        return _get_init_value((state_size, state_size), name, init_ranges)
    elseif name in ["sensory_sigma", "sensory_mu", "sensory_w"]
        return _get_init_value((_sensory_size, state_size), name, init_ranges)
    elseif name in ["erev"]
         return _erev_initializer(wiring)
    elseif name in ["sensory_erev"]
        return _sensory_erev_initializer(wiring)
    elseif name in ["sparsity_mask"]
        return abs.(wiring.adjacency_matrix)
    elseif name in ["sensory_sparsity_mask"]
        return abs.(wiring.sensory_adjacency_matrix)
    end
end

# Init all weights
function _allocate_parameters(wiring, params, init_ranges, input_mapping, output_mapping)
    println("alloc!")

    _params_keys = ["sigma", "mu", "w", "sensory_sigma", "sensory_mu", "sensory_w",
                    "erev", "sensory_erev", "gleak", "vleak", "cm", "sparsity_mask",
                    "sensory_sparsity_mask"]
    #println(fieldnames(wiring), "We're here")
    _motor_size = wiring.output_dim
    _sensory_size = wiring.input_dim
    for _key in _params_keys
        @show _key
        #_init_weight()
        params[_key] = _init_weights_and_params(wiring, _key, init_ranges)
    end
    # It is new fields !!!
    if input_mapping in ["affine", "linear"]
        params["input_w"] = ones((1, _sensory_size))
    end
    if input_mapping in ["affine"]
        params["input_b"] = zeros((1, _sensory_size))
    end
    if output_mapping in ["affine", "linear"]
        params["output_w"] = ones(_motor_size)
    end
    if output_mapping in ["affine"]
        params["output_b"] = zeros(_motor_size)
    end
    return params
end

# Additional
function _state_size(ltc::LTCCell)
    return _state_size(ltc._wiring)
end

function _map_inputs(ltc::LTCCell, inputs)
    if ltc._input_mapping in ["affine", "linear"]
        inputs = inputs .* ltc._params["input_w"] # Element-wise
    end
    if ltc._input_mapping === "affine"
        inputs = inputs .+ ltc._params["input_b"]
    end
    @assert size(inputs) === (1,2)
    return inputs
end

function _map_outputs(ltc::LTCCell, state)
    output = state
    _motor_size = ltc._wiring.output_dim

    if _motor_size < _state_size(ltc)
        println("!oups")
        #output = output[:, 0 : self.motor_size]  # slice
    end
    if ltc._output_mapping in ["affine", "linear"]
        @show size(output)
        @show size(ltc._params["output_w"])
        output = output .* reshape(ltc._params["output_w"], (1, size(ltc._params["output_w"])...)) # Element-wise
    end
    if ltc._output_mapping === "affine"
        output = output .+ reshape(ltc._params["output_w"], (1, size(ltc._params["output_b"])...)) # Element-wise
    end
    #@assert size(inputs) === (1,2)
    return output
end

function _sigmoid(v_pre, mu, sigma)
    v_pre = reshape(v_pre, (size(v_pre)...,1)) # add batch dim
    mu = reshape(mu, 1, size(mu)...) # for dims match ?

    mues = v_pre .- mu

    #x = sigma  .* mues
    x = map(x_ -> (sigma .* x_), eachslice(mues, dims=1))[end]
    x = reshape(x, 1, size(x)...) # for dims match
    return σ.(x)
end

# Test
# _sigmoid([1,2], [1,2], [1,2])

function complicated_prod(a, b; dim=1)
    # This is for keeping some dim restriction during broadcasting
    out = map(x -> a .* x, eachslice(b, dims=dim))[end]
    reshape(out, 1, size(out)...) # for dims match ?
end

function _ode_solver_(ltc::LTCCell, inputs, state, elapsed_time)
    v_pre = state
    # We can pre-compute the effects of the sensory neurons here
    println("Again we need slice(((")
    sensory_w_activation = complicated_prod(ltc._params["sensory_w"],
                                            _sigmoid(
                                                     inputs, ltc._params["sensory_mu"],
                                                     ltc._params["sensory_sigma"]))
    #@show size(sensory_w_activation)
    #@show size(ltc._params["sensory_sparsity_mask"])
    sensory_w_activation =  complicated_prod(ltc._params["sensory_sparsity_mask"],
                                             sensory_w_activation,)

    sensory_rev_activation = complicated_prod(ltc._params["sensory_erev"], sensory_w_activation)

    # Reduce over dimension 1 (=source sensory neurons)
    w_numerator_sensory = dropdims(sum(sensory_rev_activation, dims=2), dims=2)
    w_denominator_sensory =  dropdims(sum(sensory_w_activation, dims=2), dims=2)


# cm/t is loop invariant
cm_t = ltc._params["cm"] / (elapsed_time / ltc._ode_unfolds)
# Unfold the multiply ODE multiple times into one RNN step
    for t = 1:ltc._ode_unfolds
        w_activation =  complicated_prod(ltc._params["w"],
                                        _sigmoid(
                                                v_pre, ltc._params["mu"],
                                        ltc._params["sigma"]))
        w_activation = complicated_prod(ltc._params["sparsity_mask"], w_activation)

        rev_activation = complicated_prod(ltc._params["erev"], w_activation)
        # Reduce over dimension 1 (=source neurons)
        w_numerator = dropdims(sum(rev_activation, dims=2), dims=2) .+ w_numerator_sensory
        #w_numerator = sum(rev_activation, dim=2) + w_numerator_sensory
        #w_denominator = sum(w_activation, dim=2) + w_denominator_sensory
        w_denominator = dropdims(sum(w_activation, dims=2), dims=2) .+ w_denominator_sensory

        numerator = (
            reshape(cm_t, (1, size(cm_t)...)) .* v_pre
            .+ reshape(ltc._params["gleak"], (1, size(ltc._params["gleak"])...))
            .* reshape(ltc._params["vleak"], (1, size(ltc._params["vleak"])...))
            .+ w_numerator
        )
        denominator = reshape(cm_t, (1, size(cm_t)...))
             .+ reshape(ltc._params["gleak"], (1, size(ltc._params["gleak"])...))
             .+ w_denominator
        # Avoid dividing by 0
        v_pre = numerator ./ (denominator .+ ltc._epsilon)
        #@show size(v_pre)
    end
    return v_pre
end

function (ltc::LTCCell)(state, x)
    batch_size = size(x)[1] # batch dim
    seq_len = size(x)[1]
    hidden_state = zeros(batch_size, _state_size(ltc))
    t = 1 # should be loop for sequence or just "."
    inputs = x[:, t, :]
    inputs = _map_inputs(ltc, inputs)
    next_state = _ode_solver_(ltc, inputs, hidden_state, 1.0)# The 2nd arg is init state, ltc instead of RNN
    outputs = _map_outputs(ltc, next_state)
    return next_state, outputs # for Recur
end

end
