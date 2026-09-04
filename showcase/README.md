# Cosmos3-Edge local research showcase

From the repository root, run:

```shell
python -m http.server 4173 --directory showcase
```

Then open <http://localhost:4173>. The site has no build step or remote runtime
dependencies; its images and MP4 are real outputs from the local Cosmos3-Edge
INT8 experiments. The humanoid-navigation section presents the three-seed
open-loop prompt pilot honestly, including its failed task-completion result;
see [`docs/humanoid_navigation_research.md`](../docs/humanoid_navigation_research.md)
for the fine-tuning and closed-loop evaluation plan.
