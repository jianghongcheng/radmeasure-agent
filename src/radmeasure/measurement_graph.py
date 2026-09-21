"""Image-based measurement with deterministic routing and bounded VLM revision.

KEEP is research-workflow acceptance, not clinical validation. The primary angle
is always recomputed from landmarks; VLM revision changes only the second opinion.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Annotated, TypedDict
import operator
from urllib.error import HTTPError

from langgraph.graph import END, START, StateGraph

from .protocols import ProtocolRegistry
from .execution_record import _audit_value


class MeasurementState(TypedDict, total=False):
    artifact: dict
    question: str
    protocols: list[str]
    policy: dict
    landmark: dict
    vlm: dict
    initial_vlm: dict
    repairs: int
    differences: dict
    geometry_angles: dict
    decision: str
    reason: str
    result: dict
    trace: Annotated[list[dict], operator.add]


@dataclass(frozen=True)
class DiscrepancyPolicy:
    """Operator-supplied, protocol-specific thresholds; no clinical defaults."""
    keep_degrees: dict[str, float]
    stop_degrees: dict[str, float]
    max_repairs: int = 1

    def validate(self, names):
        if type(self.max_repairs) is not int or not 0 <= self.max_repairs <= 5:
            raise ValueError('max_repairs must be an integer between zero and five')
        for name in names:
            a, b = self.keep_degrees[name], self.stop_degrees[name]
            if (type(a) not in (int, float) or type(b) not in (int, float)
                    or not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b <= 90):
                raise ValueError('invalid discrepancy thresholds')


def geometry_angles(output, protocols, registry):
    """Recompute angles from axis endpoints in original pixel coordinates."""
    if output.get('quality', {}).get('passed') is not True:
        raise ValueError('landmark_quality_failed')
    if output.get('image_metadata', {}).get('contains_direct_identifiers'):
        raise ValueError('image_requires_review')
    width, height = output['image_size']
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in (width, height)):
        raise ValueError('invalid_image_dimensions')
    axes = output['axes']
    angles = {}
    for name in protocols:
        protocol = registry.get(name)
        if protocol.executor != 'acute_angle' or len(protocol.required_entities) != 2:
            raise ValueError('unsupported_geometry_executor')
        vectors = []
        for axis in protocol.required_entities:
            points = axes[axis]
            if len(points) != 2 or any(len(p) != 2 for p in points):
                raise ValueError('invalid_axis')
            for x, y in points:
                if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y))
                        or not 0 <= x < width or not 0 <= y < height):
                    raise ValueError('invalid_landmark_coordinates')
            dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
            norm = math.hypot(dx, dy)
            if norm <= 1e-6:
                raise ValueError('degenerate_axis')
            vectors.append((dx / norm, dy / norm))
        dot = abs(sum(a*b for a, b in zip(*vectors)))
        angles[name] = math.degrees(math.acos(min(1.0, dot)))
    return angles


class MeasurementGraph:
    def __init__(self, landmark, vlm, policy: DiscrepancyPolicy,
                 registry=None, checkpointer=None):
        self.landmark, self.vlm, self.policy = landmark, vlm, policy
        self.registry = registry or ProtocolRegistry()
        graph = StateGraph(MeasurementState)
        for name, fn in [('validate', self.validate), ('landmark', self.measure_landmarks),
                         ('vlm', self.estimate), ('controller', self.control),
                         ('repair', self.repair), ('finalize', self.finalize),
                         ('human_review', self.finalize)]:
            graph.add_node(name, fn)
        graph.add_edge(START, 'validate')
        # Fan out only after validation, then wait for both branches.
        graph.add_conditional_edges('validate',
            lambda s: ['human_review'] if s.get('decision') == 'STOP' else ['landmark', 'vlm'],
            ['human_review', 'landmark', 'vlm'])
        graph.add_edge(['landmark', 'vlm'], 'controller')
        graph.add_conditional_edges('controller', lambda s: {'KEEP': 'finalize', 'REPAIR': 'repair', 'STOP': 'human_review'}[s['decision']])
        graph.add_edge('repair', 'controller')
        graph.add_edge('finalize', END)
        graph.add_edge('human_review', END)
        self.graph = graph.compile(checkpointer=checkpointer)

    def validate(self, state):
        try:
            names = state['protocols']
            if not names or len(names) != len(set(names)):
                raise ValueError('invalid_protocol_selection')
            for name in names:
                self.registry.get(name)
                self.registry.validate_tools(name, ('landmark_detector', 'geometry_executor'))
            self.policy.validate(names)
            if not state['artifact'].get('sha256') or not state['artifact'].get('path'):
                raise ValueError('missing_image_artifact')
            policy = {'keep_degrees': self.policy.keep_degrees, 'stop_degrees': self.policy.stop_degrees,
                      'max_repairs': self.policy.max_repairs,
                      'protocols': [self.registry.get(n).to_dict() for n in names]}
            return {'policy': policy, 'trace': [{'step': 'validate', 'policy': policy}]}
        except (KeyError, ValueError, TypeError) as exc:
            return {'decision': 'STOP', 'reason': 'invalid_request_or_policy',
                    'trace': [{'step': 'validate', 'error_type': type(exc).__name__}]}

    def measure_landmarks(self, state):
        try:
            value = self.landmark(state['artifact'])
        except Exception as exc:
            value = {'error_type': type(exc).__name__}
        return {'landmark': value, 'trace': [{'step': 'landmark', 'output': value}]}

    def _estimate(self, state, feedback=None):
        # No landmark output is exposed on the first pass. Repair feedback has no
        # target angle: the model must inspect the image rather than copy theta1.
        try:
            value = self.vlm.estimate(state['artifact'], state['question'],
                [self.registry.get(n).to_dict() for n in state['protocols']], feedback)
        except HTTPError as exc:
            reason = 'provider_authentication_failed' if exc.code in (401, 403) else 'provider_rate_limited' if exc.code == 429 else 'provider_request_failed'
            value = {'error_type': type(exc).__name__, 'error_reason': reason}
        except Exception as exc:
            value = {'error_type': type(exc).__name__}
        return value

    def estimate(self, state):
        value = self._estimate(state)
        return {'vlm': value, 'initial_vlm': value,
                'trace': [{'step': 'vlm_initial', 'output': value}]}

    def control(self, state):
        try:
            angles = geometry_angles(state['landmark'], state['protocols'], self.registry)
            value = state['vlm']
            if value.get('status') != 'ok' or set(value['angles']) != set(angles):
                raise ValueError('invalid_or_unresolved_vlm_estimate')
            for name, angle in value['angles'].items():
                if type(angle) not in (int, float) or not math.isfinite(angle) or not 0 <= angle <= 90:
                    raise ValueError('invalid_vlm_angle')
                if not isinstance(value.get('evidence', {}).get(name), str) or not value['evidence'][name].strip():
                    raise ValueError('missing_vlm_evidence')
            differences = {n: abs(angles[n] - value['angles'][n]) for n in angles}
            if any(differences[n] > self.policy.stop_degrees[n] for n in angles):
                decision, reason = 'STOP', 'large_disagreement'
            elif all(differences[n] <= self.policy.keep_degrees[n] for n in angles):
                decision, reason = 'KEEP', 'checks_passed_and_small_disagreement'
            elif state.get('repairs', 0) >= self.policy.max_repairs:
                decision, reason = 'STOP', 'repair_budget_exhausted'
            elif any('vlm_reestimate' not in self.registry.get(n).repair_actions for n in angles):
                decision, reason = 'STOP', 'repair_not_authorized'
            else:
                for name in angles:
                    self.registry.validate_tools(name, ('vlm_reestimate',))
                decision, reason = 'REPAIR', 'moderate_disagreement'
            return {'geometry_angles': angles, 'differences': differences,
                    'decision': decision, 'reason': reason,
                    'trace': [{'step': 'controller', 'decision': decision, 'reason': reason,
                               'differences': differences, 'repairs': state.get('repairs', 0)}]}
        except (KeyError, ValueError, TypeError, IndexError, AttributeError) as exc:
            error_reason = (state.get('vlm') or {}).get('error_reason')
            reason = error_reason if error_reason in {'provider_authentication_failed', 'provider_rate_limited', 'provider_request_failed'} else 'measurement_checks_failed'
            return {'decision': 'STOP', 'reason': reason,
                    'trace': [{'step': 'controller', 'decision': 'STOP', 'error_type': type(exc).__name__}]}

    def repair(self, state):
        for name in state['protocols']:
            self.registry.validate_tools(name, ('vlm_reestimate',))
        feedback = {'reason': 'Cross-model disagreement exceeds the configured keep tolerance.',
                    'protocols_to_recheck': [n for n, d in state['differences'].items() if d > self.policy.keep_degrees[n]],
                    'previous_estimate': state['vlm'],
                    'instruction': 'Inspect the image and anatomical axes again. Do not invent agreement. Return unresolved if uncertain.'}
        value = self._estimate(state, feedback)
        return {'vlm': value, 'repairs': state.get('repairs', 0) + 1,
                'trace': [{'step': 'repair', 'tool': 'vlm_reestimate', 'feedback': feedback, 'output': value}]}

    def finalize(self, state):
        keep = state['decision'] == 'KEEP'
        result = {'measurements': state.get('geometry_angles', {}) if keep else {},
                  'candidate_measurements': state.get('geometry_angles', {}),
                  'measurement_source': 'landmark_geometry', 'initial_vlm': state.get('initial_vlm'),
                  'final_vlm': state.get('vlm'), 'landmark': state.get('landmark'),
                  'differences': state.get('differences', {}), 'repair_attempts': state.get('repairs', 0),
                  'routing': {'decision': state['decision'], 'reason': state['reason']},
                  'agent_trajectory': state.get('trace', []), 'policy': state.get('policy'),
                  'provenance': {'mode': 'dual_path_langgraph', 'clinical_use': False,
                                 'repair_scope': 'vlm_second_opinion_only'}}
        return {'result': _audit_value(result)}

    def run(self, artifact, question, protocols, *, config=None):
        return self.graph.invoke({'artifact': artifact, 'question': question,
            'protocols': list(protocols), 'repairs': 0, 'trace': []}, config=config)['result']
