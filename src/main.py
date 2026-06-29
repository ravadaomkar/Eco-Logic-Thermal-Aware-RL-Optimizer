"""
eco-logic/src/main.py
Entry point for the Eco-Logic Thermal-Aware RL Optimizer.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ecologic.log"),
    ],
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eco-Logic: Thermal-Aware RL Optimizer for Dell PowerCool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────
    train_p = subparsers.add_parser("train", help="Run RL training loop")
    train_p.add_argument("--episodes",    type=int,   default=200)
    train_p.add_argument("--cooling",     type=float, default=0.75,        help="Cooling budget [0–1]")
    train_p.add_argument("--agent",       type=str,   default="qlearning", choices=["qlearning", "dqn"])
    train_p.add_argument("--save-every",  type=int,   default=50)
    train_p.add_argument("--eval-every",  type=int,   default=25)
    train_p.add_argument("--db",          type=str,   default=None,        help="MySQL URL")
    train_p.add_argument("--prometheus",  action="store_true")
    train_p.add_argument("--seed",        type=int,   default=42)

    # ── eval ───────────────────────────────────────────────────────────
    eval_p = subparsers.add_parser("eval", help="Evaluate a saved agent")
    eval_p.add_argument("--checkpoint",   type=str,   required=True)
    eval_p.add_argument("--agent",        type=str,   default="qlearning", choices=["qlearning", "dqn"])
    eval_p.add_argument("--episodes",     type=int,   default=20)
    eval_p.add_argument("--cooling",      type=float, default=0.75)

    # ── twin ───────────────────────────────────────────────────────────
    twin_p = subparsers.add_parser("twin", help="Train and evaluate the digital twin")
    twin_p.add_argument("--data",         type=str,   default="data/rack_telemetry_sample.csv")
    twin_p.add_argument("--epochs",       type=int,   default=30)
    twin_p.add_argument("--save",         type=str,   default="checkpoints/twin.pt")

    # ── serve ──────────────────────────────────────────────────────────
    serve_p = subparsers.add_parser("serve", help="Serve the live dashboard")
    serve_p.add_argument("--port",        type=int,   default=8080)

    return parser.parse_args()


def cmd_train(args):
    from src.rl.train import train
    history = train(
        n_episodes=args.episodes,
        cooling_budget=args.cooling,
        agent_type=args.agent,
        save_every=args.save_every,
        eval_every=args.eval_every,
        db_url=args.db,
        push_prometheus=args.prometheus,
        seed=args.seed,
    )
    logger.info(f"Training complete. {len(history)} episodes logged.")


def cmd_eval(args):
    from src.rl.environment import DataCenterEnv
    from src.rl.agent import QLearningAgent
    import numpy as np

    env   = DataCenterEnv(cooling_budget=args.cooling, render_mode="human")
    agent = QLearningAgent(n_actions=env.action_space.n)
    agent.load(args.checkpoint)
    agent.epsilon = 0.0   # pure exploitation

    rewards, pues = [], []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset(seed=ep)
        total_r = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            done = term or trunc
        rewards.append(total_r)
        pues.append(info["pue"])
        logger.info(f"Eval ep {ep}: R={total_r:+.1f}  PUE={info['pue']:.3f}")

    logger.info(f"Eval summary | AvgR={np.mean(rewards):+.1f} | AvgPUE={np.mean(pues):.3f}")


def cmd_twin(args):
    try:
        import pandas as pd
        import numpy as np
        from src.digital_twin.simulator import ThermalTwinModel

        df    = pd.read_csv(args.data)
        model = ThermalTwinModel()
        logger.info("Digital twin training not fully automated in CLI; "
                    "see notebooks/02_thermal_twin_evaluation.ipynb")
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")


def cmd_serve(args):
    import http.server
    import os

    dashboard = Path("dashboard/index.html")
    if not dashboard.exists():
        logger.error("dashboard/index.html not found")
        return

    os.chdir("dashboard")
    handler = http.server.SimpleHTTPRequestHandler
    with http.server.HTTPServer(("", args.port), handler) as httpd:
        logger.info(f"Dashboard live at http://localhost:{args.port}")
        httpd.serve_forever()


if __name__ == "__main__":
    args = parse_args()
    commands = {
        "train": cmd_train,
        "eval":  cmd_eval,
        "twin":  cmd_twin,
        "serve": cmd_serve,
    }
    commands[args.command](args)
