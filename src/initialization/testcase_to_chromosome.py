import random
from MutlChromosome_v1 import MutlChromosome

class TestCaseToChromosome:
    def __init__(self):
        self.action_mapping = {
            'accelerate': 0,
            'decelerate': 1,
            'changeLane': 2,
            'stop': 3,
            'idle': 4
        }

    def convert(self, testcase, bounds, time_size, pools):
        """Convert a TestCase to MutlChromosome format"""
        # Count number of NPCs (excluding ego)
        npc_count = sum(1 for stmt in testcase.constructor_statements if stmt.assignee != 'ego')

        # Create MutlChromosome instance
        chromosome = MutlChromosome(bounds, npc_count, time_size, pools)

        # Convert NPC initializations and actions
        npc_index = 0
        for stmt in testcase.constructor_statements:
            if stmt.assignee == 'ego':
                continue

            # Handle NPC initialization
            x = random.uniform(stmt.arg_bounds.get('offset', [0])[0],
                             stmt.arg_bounds.get('offset', [10])[1])
            z = random.uniform(-1, 1)  # Lane position
            n_v = stmt.arg_bounds.get('initial_speed', [0])[0]

            # Initialize first time step for this NPC
            chromosome.scenario[npc_index][0] = self._create_motif_gene(
                x, z, n_v, self.action_mapping['idle']
            )

            npc_index += 1

        # Convert method calls to actions
        time_step = 1
        for stmt in testcase.method_statements:
            if time_step >= time_size:
                break

            npc_index = int(stmt.callee.replace('vehicle', '')) - 1
            action_type = self.action_mapping.get(stmt.method_name, 4)

            # Create gene based on action
            chromosome.scenario[npc_index][time_step] = self._create_motif_gene(
                chromosome.scenario[npc_index][time_step-1][2],  # Previous x
                chromosome.scenario[npc_index][time_step-1][3],  # Previous z
                stmt.arg_bounds.get('target_speed', [0])[0],
                action_type
            )

            time_step += 1

        return chromosome

    def _create_motif_gene(self, x, z, speed, action):
        """Create a motif gene with the given parameters"""
        v = {
            "decelerate": random.uniform(0.1, 0.3),
            "accalare": random.uniform(0.1, 0.3),
            "stop": 0,
            "lanechangspeed": random.uniform(0.1, 0.3)
        }
        return [v, action, x, z, speed, random.randint(0, 5)]