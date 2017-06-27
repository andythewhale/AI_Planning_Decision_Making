from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph
from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list
            NOTE: Alright I'm sort of lost on how to set this up, initially.
            NOTE: I think the best thing to do is to log my thinking process in finding a solution.
            -So we get to import cargos, planes, airports, the initial state and the initial goal.
            -So those are given and I have to do something with them.
            -I'm going to also need to check out a bunch of the utility code that's given.
            -I'm definitely going to have to understand symbolic code better so I'll go through som resources. (Ch11)



            :return: list of Action objects
            """
            loads = []
            # TODO create all load ground actions from the domain Load action
            # I'm going to use the order least to greatest things, because I think it's the most efficient.
            # So for each airport in our problem
            for a in self.airports:
                # For each airplane in our problem
                for p in self.planes:
                    # For each cargo in our problem
                    for c in self.cargos:
                        # It took me a million years to find expr in utils.
                        # we need to make our preconditions first:
                        # Our preconditions are the airplane and cargo being at this airport
                        precond_pos = [expr("At({}, {})".format(c, a)), expr("At({}, {})".format(p, a))]
                        # There are no negating preconditions
                        precond_neg = []
                        # The effect is to add the cargo to the plane and remove the cargo from the airport
                        effect_add = [expr("In({}, {})".format(c, p))]
                        effect_rem = [expr("At({}, {})".format(c, a))]

                        # Now we need to do our action.
                        load_action = Action(expr("Load({}, {}, {})".format(c, p, a)),
                                             [precond_pos, precond_neg], [effect_add, effect_rem])

                        # append our action to our loads
                        loads.append(load_action)
            # return loads
            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            # TODO create all Unload ground actions from the domain Unload action
            # Literally just the opposite of loads.
            # For each airport in our problem
            for a in self.airports:
                # For each airplane in our problem
                for p in self.planes:
                    # For each cargo in our problem
                    for c in self.cargos:
                        # It took me a million years to find expr in utils.
                        # we need to make our preconditions first:
                        # Our preconditions are the cargo being in the plane and thd plane being at the airport
                        precond_pos = [expr("In({}, {})".format(c, p)), expr("At({}, {})".format(p, a))]
                        # There are no negating preconditions
                        precond_neg = []
                        # The effect is to add the cargo to the plane and remove the cargo from the airport
                        effect_add = [expr("At({}, {})".format(c, a))]
                        effect_rem = [expr("In({}, {})".format(c, p))]

                        # Now we need to do our action.
                        unload_action = Action(expr("Unload({}, {}, {})".format(c, p, a)),
                                             [precond_pos, precond_neg], [effect_add, effect_rem])

                        # append our action to our loads
                        unloads.append(unload_action)
            # return unloads
            return unloads


        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr))]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        # TODO implement
        possible_actions = []

        # So in each current state we need to implement the possible actions for that state.
        # I'm literally copying what we used for the cake problem. But I'll annotate it to understand it more.
        # So KB is just a  knowledge base, and PropKB is just a subclass of that knowledge base.

        # So first we call our subclass of KB. It uses propositional logic.
        kb = PropKB()

        # So we want to add the current state (tell) it to our kb.
        # As you can see, goal test does this as well.
        # pos_sentence() is something that occurs in the Fluent State Class.
        # It returns a conjunctive sentence of the positive fluent.
        # self.pos is the pos_list of actions that fluent state is initialized with.
        kb.tell(decode_state(state, self.state_map).pos_sentence())

        # for actions in the self.actions_list:
        for action in self.actions_list:

            # check if it's possible:

            # Assume possible initially:
            is_possible = True

            # For each clause in the action.positive preconditions:
            for clause in action.precond_pos:

                # Check if the clause is in kb.clauses.
                # If it isn't it is impossible.
                # clauses is defined in lp_utils. It appends the pos and negative list to the clauses list.
                if clause not in kb.clauses:

                    #Can't be possible if it's not a clause of clauses:
                    is_possible = False

            #Same as last for loop except for the negative precondition, and if it's negative then it's not possible.
            for clause in action.precond_neg:

                #If it's false then it's not possible.
                if clause in kb.clauses:
                    is_possible = False

                # If none of previous conditions are true then it is true.
            if is_possible:
                possible_actions.append(action)

            #And we do this for each action in the list and we return a list of pos actions.
        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        # TODO implement
        # Again this is just from the have cake example. Which is fine. I'm just going to annotate it to understand it better

        #Start with a new blank state. One for pos, one for neg.
        new_state = FluentState([], [])
        #Let's figure out what our old state was.
        old_state = decode_state(state, self.state_map)

        #For each fluent in the old state positives:
        for fluent in old_state.pos:
            # If the state isn't available in the removed states then it is part of the new state.
            if fluent not in action.effect_rem:
                new_state.pos.append(fluent)

        # For fluent in the add actions:
        for fluent in action.effect_add:
            #For each fluent in the added state. Add it to the new state.
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)

        #Do the same thing above but for negatives.
        for fluent in old_state.neg:
            #For each fluent not in the added effect sate. Add it to the neg.
            if fluent not in action.effect_add:
                    new_state.neg.append(fluent)

        #For the fluent in the removed effects
        for fluent in action.effect_rem:
            #If the fluent is not already in the new neg state then add it.
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)

        #return the new state.
        return encode_state(new_state, self.state_map)

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        # TODO implement (see Russell-Norvig Ed-3 10.2.3  or Russell-Norvig Ed-2 11.2)
        # So this is just goal_test as a counter. We just ignore our initial state and count

        count = 0
        # So first we call our subclass of KB. It uses propositional logic.
        kb = PropKB()

        # So we want to add the current state (tell) it to our kb.
        # As you can see, goal test does this as well.
        # pos_sentence() is something that occurs in the Fluent State Class.
        # It returns a conjunctive sentence of the positive fluent.
        # self.pos is the pos_list of actions that fluent state is initialized with.
        # NOTE: There is a hint here. Node is called as an input.
        # This is important.
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())

        #Literally copied from example_have_cake.py
        #Just with a counter.
        for clause in self.goal:
            if clause not in kb.clauses:
                count = count + 1
        return count


def air_cargo_p1() -> AirCargoProblem:

    """
    Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(P1, SFO) ∧ At(P2, JFK)
	∧ Cargo(C1) ∧ Cargo(C2) ∧ Plane(P1) ∧ Plane(P2)
	∧ Airport(JFK) ∧ Airport(SFO))

    Goal(At(C1, JFK) ∧ At(C2, SFO))
    """

    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    # TODO implement Problem 2 definition
    # #This part is pretty straightforward I"m just picking the positive and negative states
    #I'm copying the state above so my logic program knows what to do to get to the goal.

    """
    It'll help to have the state and goal here:

    Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(P1, SFO)
    ∧ At(P2, JFK) ∧ At(P3, ATL) ∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3)
    ∧ Plane(P1) ∧ Plane(P2) ∧ Plane(P3) ∧ Airport(JFK) ∧ Airport(SFO)
    ∧ Airport(ATL))

    Goal(At(C1, JFK) ∧ At(C2, SFO) ∧ At(C3, SFO))
    """

    #Here's all the objects we'll use:
    airports = ['JFK', 'SFO', 'ATL']
    planes = ['P1', 'P2', 'P3']
    cargos = ['C1', 'C2', 'C3']

    #Here's where we start at:
    pos = [expr('At(C1, SFO)'), expr('At(C2, JFK)'), expr('At(C3, ATL)'),
           expr('At(P1, SFO)'), expr('At(P2, JFK)'), expr('At(P3, ATL)')]


    #Here's where we don't want to go, literally we call every other state.
    neg = [expr('At(C1, ATL)'), expr('At(C2, SFO)'), expr('At(C3, JFK)'),
           expr('At(P1, ATL)'), expr('At(P2, SFO)'), expr('At(P3, JFK)'),
           expr('At(C1, JFK)'), expr('At(C2, ATL)'), expr('At(C3, SFO)'),
           expr('At(P1, JFK)'), expr('At(P2, ATL)'), expr('At(P3, SFO)'),
           expr('In(C1, P1)'), expr('In(C2, P2)'), expr('In(C3, P3)'),
           expr('In(C1, P2)'), expr('In(C2, P3)'), expr('In(P3, P1)'),
           expr('In(C1, P3)'), expr('In(C2, P1)'), expr('In(C3, P2)')]

    #Call the FluentState as our initial state (like p1)
    init = FluentState(pos, neg)

    #This is the goal.
    goal = [expr('At(C1, JFK)'), expr('At(C2, SFO,)'), expr('At(C3, SFO)')]

    #return and call function.
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    # TODO implement Problem 3 definition
    # #This part is pretty straightforward I"m just picking the positive and negative states
    #I'm copying the state above so my logic program knows what to do to get to the goal.

    """
    Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(C4, ORD)
	∧ At(P1, SFO) ∧ At(P2, JFK) ∧ Cargo(C1) ∧ Cargo(C2)
	∧ Cargo(C3) ∧ Cargo(C4) ∧ Plane(P1) ∧ Plane(P2)
	∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL) ∧ Airport(ORD))

    Goal(At(C1, JFK) ∧ At(C3, JFK) ∧ At(C2, SFO) ∧ At(C4, SFO))
    """

    #Here's all the objects we'll use:
    airports = ['ORD', 'SFO', 'ATL', 'JFK']
    planes = ['P1', 'P2']
    cargos = ['C1', 'C2', 'C3', 'C4']

    #Here's where we start at:
    pos = [expr('At(C1, SFO)'), expr('At(C2, JFK)'), expr('At(C3, ATL)'),
           expr('At(C4, ORD)'), expr('At(P1, SFO)'), expr('At(P2, JFK)')]


    #Here's where we don't want to go, literally we call every other state.
    neg = [expr('At(P1, ATL)'), expr('At(P1, ORD)'), expr('At(P1, JFK)'),
           expr('At(P2, ATL)'), expr('At(P2, SFO)'), expr('At(P2, ORD)'),
           expr('At(C1, JFK)'), expr('At(C1, ATL)'), expr('At(C1, ORD)'),
           expr('In(C1, P1)'), expr('In(C1, P2)'), expr('At(C2, SFO)'),
           expr('At(C2, ORD)'), expr('At(C2, ATL)'), expr('In(C2, P1)'),
           expr('In(C2, P2)'), expr('In(C3, P1)'), expr('In(C3, P2)'),
           expr('At(C3, JFK)'), expr('At(C3, ORD)'), expr('At(C3, SFO)'),
           expr('At(C4, JFK)'), expr('At(C4, SFO)'), expr('At(C4, ORD)'),
           expr('In(C4, P1)'), expr('In(C4, P2)')]

    #Call the FluentState as our initial state (like p1)
    init = FluentState(pos, neg)

    #This is the goal.
    goal = [expr('At(C1, JFK)'), expr('At(C3, JFK)'), expr('At(C2, SFO)'), expr('At(C4, SFO)')]

    #return and call function.
    return AirCargoProblem(cargos, planes, airports, init, goal)
