module DDPG

using Flux, Flux.Optimise
import Flux.params
using Distributions
using Statistics
using Conda
using PyCall
using Parameters 
using UnPack
using MLUtils
# import Base.push!


export  agent, HyperParameter


@with_kw mutable struct EnvParameter
    # Dimensions
    ## Actions
    action_size::Int =                      1
    action_bound::Float32 =                 1.f0
    action_bound_high::Array{Float32} =     [1.f0]
    action_bound_low::Array{Float32} =      [-1.f0]
    ## States
    state_size::Int =                       1
    state_bound_high::Array{Float32} =      [1.f0]
    state_bound_low::Array{Float32} =       [1.f0]
end

@with_kw mutable struct HyperParameter
    # Buffer size
    buffer_size::Int =                      1000000
    # Exploration
    expl_noise::Float32 =                   0.2f0
    noise_clip::Float32 =                   1.f0
    # Training Metrics
    training_episodes::Int =                300
    maximum_episode_length::Int =           1000
    train_start:: Int =                     10
    batch_size::Int =                       64
    # Metrics
    episode_reward::Array{Float32} =        []
    critic_loss::Array{Float32} =           [0.f0]
    actor_loss::Array{Float32} =            [0.f0]
    episode_steps::Array{Int} =             []
    # Discount
    γ::Float32 =                            0.99f0
    # Learning Rates
    critic_η::Float64 =                     0.001
    actor_η::Float64 =                      0.001
    # Agents
    store_frequency::Int =                  100
    trained_agents =                        []
end

        
function setCritic(state_size, action_size)

    return Chain(Dense(state_size + action_size, 400, relu),
                    Dense(400, 300, relu),
                    Dense(300, 1))
    # return Chain(Dense(state_size + action_size, 4, relu),
    #                 Dense(4, 3, relu),
    #                 Dense(3, 1))
                    
end


function setActor(state_size, action_size)

    # return Chain(Dense(state_size, 4, relu),
    #                 Dense(4, 3, relu),
    #                 Dense(3, action_size, tanh))
    return Chain(Dense(state_size, 400, relu),
                    Dense(400, 300, relu),
                    Dense(300, action_size, tanh))

end



# Define the experience replay buffer
mutable struct ReplayBuffer
    capacity::Int
    memory::Vector{Tuple{Vector{Float32}, Vector{Float32}, Float32, Vector{Float32}, Bool}}
    pos::Int
end

# outer constructor for the Replay Buffer
function ReplayBuffer(capacity::Int)
    memory = []
    return ReplayBuffer(capacity, memory, 1)
end


function remember(buffer::ReplayBuffer, state, action, reward, next_state, done)
    if length(buffer.memory) < buffer.capacity
        push!(buffer.memory, (state, action, reward, next_state, done))
    else
        buffer.memory[buffer.pos] = (state, action, reward, next_state, done)
    end
    buffer.pos = mod1(buffer.pos + 1, buffer.capacity)
end


function sample(buffer::ReplayBuffer, batch_size::Int)
    batch = rand(buffer.memory, batch_size)
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for (s, a, r, ns, d) in batch
        push!(states, s)
        push!(actions, a)
        push!(rewards, r)
        push!(next_states, ns)
        push!(dones, d)
    end
    return hcat(states...), hcat(actions...), rewards, hcat(next_states...), dones
end


# Define the action, actor_loss, and critic_loss functions
function action(model, state, train, ep, hp)
    if train
        a = model(state) .+ clamp.(rand(Normal{Float32}(0.f0, hp.expl_noise), size(ep.action_size)), -hp.noise_clip, hp.noise_clip)
        return clamp.(a, ep.action_bound_low, ep.action_bound_high)
    else
        return model(state)
    end
end



function soft_update!(target_model, main_model, τ)
    for (target_param, main_param) in zip(Flux.params(target_model), Flux.params(main_model))
        target_param .= τ * main_param .+ (1 - τ) * target_param
    end
end



function verify_update(target_model, main_model)
    for (i, (target_param, main_param)) in enumerate(zip(Flux.params(target_model), Flux.params(main_model)))
        diff = main_param - target_param
        println("Difference for parameter $i:")
        println(diff)
    end
end



function train_step!(S, A, R, S´, T, μθ, μθ´, Qϕ, Qϕ´, hp::HyperParameter)

    Y = R' .+ hp.γ * (1 .- T)' .* Qϕ´(vcat(S´, μθ´(S)))   

    # Works
    #@show Flux.params(Qϕ)[1]
    dϕ = Flux.gradient(m -> Flux.Losses.mse(m(vcat(S, A)), Y), Qϕ)
    Flux.update!(opt_critic, Qϕ, dϕ[1])
    #@show Flux.params(Qϕ)[1]
    
    push!(hp.critic_loss, Flux.Losses.mse(Qϕ(vcat(S, A)), Y))
    
    #actor
    dθ = Flux.gradient(m -> -mean(Qϕ(vcat(S, m(S)))), μθ)
    Flux.update!(opt_actor, μθ, dθ[1])
    
    push!(hp.actor_loss, -mean(Qϕ(vcat(S, μθ(S)))))
    
    #@show Flux.params(Qϕ´)[1]
    soft_update!(Qϕ´, Qϕ, 0.005)
    soft_update!(μθ´, μθ, 0.005)
    #@show Flux.params(Qϕ´)[1]

    #verify_update(Qϕ´, Qϕ)

end


function agent(environment, hyperParams::HyperParameter)
    println("Hello people I am here")

    gym = pyimport("gym")
    env = gym.make(environment)
    envParams = EnvParameter()

    # Reset Parameters
    ## ActionenvP
    envParams.action_size =        env.action_space.shape[1]
    envParams.action_bound =       env.action_space.high[1]
    envParams.action_bound_high =  env.action_space.high
    envParams.action_bound_low =   env.action_space.low
    ## States
    envParams.state_size =         env.observation_space.shape[1]
    envParams.state_bound_high =   env.observation_space.high
    envParams.state_bound_low =    env.observation_space.low


    episode = 1

    μθ = setActor(envParams.state_size, envParams.action_size)
    μθ´= deepcopy(μθ)
    Qϕ = setCritic(envParams.state_size, envParams.action_size)
    Qϕ´= deepcopy(Qϕ)

    global opt_critic = Flux.setup(Flux.Optimise.Adam(hyperParams.critic_η), Qϕ)
    global opt_actor = Flux.setup(Flux.Optimise.Adam(hyperParams.actor_η), μθ)
    
    buffer = ReplayBuffer(hyperParams.buffer_size)

    while episode ≤ hyperParams.training_episodes

        frames = 0
        s = env.reset()[1]
        episode_rewards = 0
        t = false
        
        for step in 1:hyperParams.maximum_episode_length
            
            a = action(μθ, s, true, envParams, hyperParams)
            s´, r, t, _ = env.step(a)

            episode_rewards += r
            
            remember(buffer, s, a, r, s´, t)

            if episode > hyperParams.train_start

                S, A, R, S´, T = sample(buffer, hyperParams.batch_size)
                train_step!(S, A, R, S´, T, μθ, μθ´, Qϕ, Qϕ´, hyperParams)
                
            end

            
            s = s´
            frames += 1
            
            if t
                env.close()
                break
            end
            
        end

        
        if episode % hyperParams.store_frequency == 0
            push!(hyperParams.trained_agents, μθ)
        end


        push!(hyperParams.episode_steps, frames)
        push!(hyperParams.episode_reward, episode_rewards)

        println("Episode: $episode | Cumulative Reward: $(round(episode_rewards, digits=2)) | Critic Loss: $(hyperParams.critic_loss[end]) | Actor Loss: $(hyperParams.actor_loss[end]) | Steps: $(frames)")
        
        episode += 1
    
    end
    
    return hyperParams
    
end


# Works
# agent(DDPG(), "BipedalWalker-v3", HyperParameter(expl_noise=0.1f0, noise_clip=0.3f0, training_episodes=10000, maximum_episode_length=3000, train_start=20, batch_size=128, critic_η=0.0001, actor_η=0.0001))
# hp = agent(DDPG(), "BipedalWalker-v3", HyperParameter(expl_noise=0.2f0, noise_clip=0.5f0, training_episodes=200, maximum_episode_length=1000, train_start=20, batch_size=64, store_frequency=20))
# hp = agent("Pendulum-v1", HyperParameter(expl_noise=0.2f0, noise_clip=0.5f0, training_episodes=200, maximum_episode_length=1000, train_start=20, batch_size=64, store_frequency=20))




end # module DDPG
