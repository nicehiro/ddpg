from puck_world.envs.single_agent.puckworld_continuous import PuckWorld
from ddpg_pytorch.agent import Trainer as DDPGAgent

if __name__ == '__main__':
    env = PuckWorld()
    # env = gym.make('Pendulum-v0')
    agent = DDPGAgent(env)
    agent.is_trainning = False
    agent.load_weights('ddpg')
    for epi in range(1000):
        s = env.reset()
        steps = 0
        for _ in range(300):
            steps += 1
            env.render()
            a = agent.act(s)
            s_, r, done, _ = env.step(a)
            s = s_
            if done:
                break
        print('Episode: %d\tSteps: %d' % (epi, steps))
