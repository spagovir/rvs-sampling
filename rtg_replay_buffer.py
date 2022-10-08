# %%
from dopamine.dopamine.replay_memory import circular_replay_buffer

class RtGBuffer(circular_replay_buffer.OutOfGraphReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cumulative_discount_vector[0] = 0

# %%
rb = RtGBuffer(
    observation_shape=(84,84), 
    stack_size = 4, 
    replay_capacity = 100000, 
    batch_size = 32,
    update_horizon=200)

rb2 = RtGBuffer(
    observation_shape=(84,84), 
    stack_size = 4, 
    replay_capacity = 100000, 
    batch_size = 32,
    update_horizon=1)
    
    

# %%
rb.load("./atari_data/dqn/SpaceInvaders/1/replay_logs/", 0 )
rb2.load("./atari_data/dqn/SpaceInvaders/1/replay_logs/", 0)
# %%
i=0
while True:
    states, ac, ret, next_states, _, _, terminal, indices = rb.sample_transition_batch(1,[i])
    _, _, _, _, _, next_reward, terminal, indices = rb2.sample_transition_batch(1, [i])
    ret = ret.item()
    next_reward = next_reward.item()
    terminal=terminal.item()
    print(f"{ret=}, {next_reward=}, {terminal=}\n")
    if terminal:
        break
    i += 1
# %%
