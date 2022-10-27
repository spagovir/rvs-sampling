# %%
from dopamine.dopamine.replay_memory import circular_replay_buffer
from tqdm import tqdm

# %%
rb = circular_replay_buffer.OutOfGraphReplayBuffer(
    observation_shape=(84,84), 
    stack_size = 4, 
    replay_capacity = 1000000, 
    batch_size = 32,
    update_horizon=1,
    gamma = 0.9999)
# %%
rb.load("./atari_data/dqn/SpaceInvaders/1/replay_logs/", 40 )
# %%
min_id = max(rb.cursor() - rb._replay_capacity, 0) + rb._stack_size - 1
max_id = rb.cursor() - 1
# %%
max_reward = 0
max_length = 0
current_reward = 0
current_length = 0
for i in tqdm(range(min_id,max_id)):
    _, _, _, _, _, next_reward, terminal, indices = rb.sample_transition_batch(1, [i])
    if terminal: 
        current_reward = 0
        current_length = 0
    current_reward += next_reward
    current_length += 1
    max_reward = max(max_reward,current_reward)
    max_length = max(max_length, current_length)
# %%
max_reward
# %%
max_length
# %%
