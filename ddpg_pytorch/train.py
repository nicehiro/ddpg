from puck_world.envs.single_agent.puckworld_continuous import PuckWorld
from ddpg_pytorch.agent import Trainer as DDPGAgent
from utils import writer

if __name__ == '__main__':
    env = PuckWorld()
    # env = gym.make('Pendulum-v0')
    agent = DDPGAgent(env, retrain=False)
    steps = 0
    display = False
    for epi in range(10000):
        s = env.reset()
        t = 0
        for _ in range(500):
            steps += 1
            t += 1
            if display:
                env.render()
            a = agent.act(s)
            s_, r, d, _ = env.step(a)
            agent.replay_buffer.append(s, a, r, d, s_)
            s = s_
            if d:
                break
            agent.optimize()
        print('Episode %i, Steps: %i' % (epi, t))
        if epi % 100 == 0:
            agent.save_model('ddpg-obs')

