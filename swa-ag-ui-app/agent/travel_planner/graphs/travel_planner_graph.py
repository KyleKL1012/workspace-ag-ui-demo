from __future__ import annotations

from langgraph.graph import StateGraph

from travel_planner.helpers.logs import get_logger
from travel_planner.models.router_models import Routes
from travel_planner.models.state import TravelPlannerState
from travel_planner.nodes.node_factory import NodeFactory


class TravelPlannerGraph:
    """
    Builds and compiles the LangGraph pipeline for the Gen-AI travel planner.
    """

    def __init__(self, node_factory: NodeFactory):
        # Create a shortcut, normally it's not a good practice but it will be used
        # frequently in this class.
        self._nf = node_factory
        self.logger = get_logger()

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    def build_graph(self) -> StateGraph:
        """Create, connect and compile the StateGraph with human-in-the-loop support."""
        graph = StateGraph(TravelPlannerState)
        graph = self._add_nodes(graph)
        graph = self._connect_edges(graph)
        return graph

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #
    def _add_nodes(self, graph: StateGraph) -> StateGraph:
        """Register each node held by NodeFactory into the graph."""
        # Router Node - entry point
        graph.add_node(self._nf.router_node.node_id, self._nf.router_node.async_run)

        # Chitchat Node
        graph.add_node(self._nf.chitchat_node.node_id, self._nf.chitchat_node.async_run)

        # Escalation Node
        graph.add_node(self._nf.escalation_node.node_id, self._nf.escalation_node.async_run)

        # Turkish Airlines Node
        graph.add_node(
            self._nf.turkish_airlines_node.node_id, self._nf.turkish_airlines_node.async_run
        )

        # Collect Trip Params Node
        graph.add_node(
            self._nf.extract_trip_params_node.node_id, self._nf.extract_trip_params_node.async_run
        )

        # Fix Trip Params Node if necessary
        graph.add_node(
            self._nf.fix_trip_params_node.node_id, self._nf.fix_trip_params_node.async_run
        )

        # Human Input Node for collecting missing information
        graph.add_node(
            self._nf.trip_params_human_input_node.node_id,
            self._nf.trip_params_human_input_node.async_run,
        )

        # LLM Trip Planner Node
        graph.add_node(
            self._nf.llm_trip_planner_node.node_id, self._nf.llm_trip_planner_node.async_run
        )

        return graph

    def _connect_edges(self, graph: StateGraph) -> StateGraph:
        """
        Wire the ordered execution flow with human-in-the-loop capability.
        """
        # Set router as entry point
        graph.set_entry_point(self._nf.router_node.node_id)

        # Add conditional edge from router based on routing decision
        graph.add_conditional_edges(
            self._nf.router_node.node_id,
            self._decide_next_route,
            {
                self._nf.extract_trip_params_node.node_id: self._nf.extract_trip_params_node.node_id,
                self._nf.chitchat_node.node_id: self._nf.chitchat_node.node_id,
                self._nf.escalation_node.node_id: self._nf.escalation_node.node_id,
                self._nf.turkish_airlines_node.node_id: self._nf.turkish_airlines_node.node_id,
            },
        )

        # Chitchat, escalation, and Turkish Airlines nodes are end points
        graph.set_finish_point(self._nf.chitchat_node.node_id)
        graph.set_finish_point(self._nf.escalation_node.node_id)
        graph.set_finish_point(self._nf.turkish_airlines_node.node_id)

        # Travel planner flow continues as before
        # Add conditional edge based on whether trip params need fixing
        graph.add_conditional_edges(
            self._nf.extract_trip_params_node.node_id,
            self._should_fix_trip_params,
            {
                self._nf.fix_trip_params_node.node_id: self._nf.fix_trip_params_node.node_id,
                self._nf.llm_trip_planner_node.node_id: self._nf.llm_trip_planner_node.node_id,
            },
        )

        # After fix attempt, go to human input for missing information
        graph.add_edge(
            self._nf.fix_trip_params_node.node_id, self._nf.trip_params_human_input_node.node_id
        )

        # After human input, go back to collect params to re-process with new info
        graph.add_edge(
            self._nf.trip_params_human_input_node.node_id, self._nf.extract_trip_params_node.node_id
        )

        graph.set_finish_point(self._nf.llm_trip_planner_node.node_id)
        return graph

    def _should_fix_trip_params(self, state: TravelPlannerState) -> str:
        """
        Decision function to determine if trip parameters need fixing.

        Args:
            state: Current state of the travel planner

        Returns:
            "fix_params" if there are missing trip parameters, "continue" otherwise
        """
        if state.missing_trip_params and len(state.missing_trip_params) > 0:
            return self._nf.fix_trip_params_node.node_id
        return self._nf.llm_trip_planner_node.node_id

    def _decide_next_route(self, state: TravelPlannerState) -> str:
        if (
            state.routing_decision is None
            or state.routing_decision.predicted_route == Routes.CHITCHAT
        ):
            # Default to chitchat if no routing decision available
            return self._nf.chitchat_node.node_id
        elif state.routing_decision.predicted_route == Routes.ESCALATION:
            return self._nf.escalation_node.node_id
        elif state.routing_decision.predicted_route == Routes.TRAVEL_PLANNER:
            return self._nf.extract_trip_params_node.node_id
        elif state.routing_decision.predicted_route == Routes.TURKISH_AIRLINES:
            return self._nf.turkish_airlines_node.node_id
        else:
            self.logger.error(
                f"Unknown routing decision: {state.routing_decision.predicted_route}Escalating"
            )
            return self._nf.escalation_node.node_id
