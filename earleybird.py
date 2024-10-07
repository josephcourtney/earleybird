import logging

from rich.console import Console
from rich.table import Table

console = Console()


class Rule:
    def __init__(self, lhs: str, rhs: list[str]):
        """Initialize a Rule with a left-hand side and a list of right-hand side symbols."""
        self.lhs = lhs  # Left-hand side non-terminal
        self.rhs = rhs  # Right-hand side symbols (list)

    def __repr__(self) -> str:
        """Return a string representation of the Rule."""
        return f"{self.lhs} -> {" ".join(self.rhs)}"


class Grammar:
    def __init__(self, rules: list[Rule], start_symbol: str):
        """Initialize a Grammar with a list of rules and a start symbol."""
        self.rules = rules
        self.start_symbol = start_symbol
        self.rules_by_lhs: dict[str, list[Rule]] = {}
        self.nonterminals: set[str] = set()
        self.terminals: set[str] = set()
        for rule in rules:
            self.rules_by_lhs.setdefault(rule.lhs, []).append(rule)
            self.nonterminals.add(rule.lhs)
            for symbol in rule.rhs:
                if symbol.isupper():
                    self.nonterminals.add(symbol)
                else:
                    self.terminals.add(symbol)


class State:
    def __init__(self, rule: Rule, dot: int, start: int, comment: str):
        """Initialize a State with a rule, dot position, start position, and a comment."""
        self.rule = rule  # Rule being processed
        self.dot = dot  # Position of the dot in RHS
        self.start = start  # Start position in the input
        self.back_pointers: list[State | str] = []  # List of possible back-pointers
        self.comment = comment  # Comment explaining how the state was added

    def next_symbol(self) -> str | None:
        """Return the next symbol after the dot, if available."""
        return self.rule.rhs[self.dot] if self.dot < len(self.rule.rhs) else None

    def is_complete(self) -> bool:
        """Check if the state is complete (dot is at the end of the rule)."""
        return self.dot >= len(self.rule.rhs)

    def __eq__(self, other: object) -> bool:
        """Check equality with another State object."""
        if not isinstance(other, State):
            return False
        return (self.rule, self.dot, self.start) == (other.rule, other.dot, other.start)

    def __hash__(self) -> int:
        """Return a hash of the State."""
        return hash((self.rule, self.dot, self.start))

    def __repr__(self) -> str:
        """Return a string representation of the State."""
        rhs = self.rule.rhs[:]
        rhs.insert(self.dot, "•")
        return f"({self.rule.lhs} -> {" ".join(rhs)})"


class Chart:
    def __init__(self, grammar: Grammar):
        """Initialize the chart with the start rules of the grammar."""
        self.chart = [{}]  # Initialize chart with one set of states
        self.add_initial_states(grammar)

    def extend(self, n: int) -> None:
        """Extend the chart to accommodate additional positions."""
        self.chart.extend([{} for _ in range(n + 1)])

    def __getitem__(self, index: int) -> dict:
        """Get the chart at a specific index."""
        return self.chart[index]

    def __setitem__(self, index: int, value: dict) -> None:
        """Set the chart at a specific index."""
        self.chart[index] = value

    def add(self, state: State, position: int, states: list[State]) -> None:
        """Add a state to the chart, updating back-pointers if necessary."""
        existing_state = self.chart[position].get(state)
        if existing_state is None:
            print(f"Adding state: {state} at position {position}")
            self.chart[position][state] = state
            states.append(state)
        else:
            existing_back_pointers = set(existing_state.back_pointers)
            for bp in state.back_pointers:
                if bp not in existing_back_pointers:
                    existing_state.back_pointers.append(bp)
                    existing_back_pointers.add(bp)

    def add_initial_states(self, grammar: Grammar) -> None:
        """Add the initial states for all rules corresponding to the start symbol."""
        initial_states = []
        for rule in grammar.rules_by_lhs.get(grammar.start_symbol, []):
            state = State(rule, 0, 0, comment="initial state")
            self.add(state, 0, initial_states)


class EarleyParser:
    def __init__(self, grammar: Grammar, *, verbose: bool = False):
        """Initialize the Earley Parser with a given grammar and verbosity setting."""
        self.grammar = grammar
        self.leo_memo: list[dict[str, State]] = []  # Memoize completed states by non-terminal
        self.nullable_nonterminals = set()
        self.verbose = verbose  # Control for debug output
        self._compute_nullable_nonterminals()
        self.chart = Chart(grammar)

        # Setup logging
        self.logger = logging.getLogger("EarleyParser")
        if verbose:
            logging.basicConfig(level=logging.INFO)

    def _log(self, message: str) -> None:
        """Log the message if verbosity is enabled."""
        self.logger.info(message)

    def _compute_nullable_nonterminals(self) -> None:
        """Precompute nullable non-terminals with indirect nullability."""
        change = True
        while change:
            change = False
            for rule in self.grammar.rules:
                if rule.lhs not in self.nullable_nonterminals and all(
                    not symbol or symbol in self.nullable_nonterminals for symbol in rule.rhs
                ):
                    self.nullable_nonterminals.add(rule.lhs)
                    change = True

    @staticmethod
    def _is_right_recursive(rule: Rule) -> bool:
        """Check if the rule is right-recursive."""
        return len(rule.rhs) > 0 and rule.rhs[-1] == rule.lhs

    def parse(self, tokens: list[str]) -> tuple[str, list] | None:
        """Parse a list of tokens using the Earley parsing algorithm."""
        self.tokens = tokens
        self.chart.extend(len(tokens))
        self.leo_memo = [{} for _ in range(len(tokens) + 1)]  # Reset memo structure for each parse

        if not tokens and self.grammar.start_symbol in self.nullable_nonterminals:
            # Special case: Handle empty input if the start symbol is nullable
            for state in self.chart[0].values():
                if state.rule.lhs == self.grammar.start_symbol and state.is_complete():
                    return self._build_parse_tree(state)
            return None

        for position in range(len(tokens) + 1):
            states = list(self.chart[position].values())
            self._process_states(states, position)

        for state in self.chart[len(tokens)].values():
            if state.rule.lhs == self.grammar.start_symbol and state.is_complete() and state.start == 0:
                return self._build_parse_tree(state)

        self._log(f"Parsing failed: No valid parse found for tokens {tokens}")
        return None

    def _process_states(self, states: list[State], position: int) -> None:
        """General method to handle predict, scan, and complete for each state."""
        for state in states:
            if state.is_complete():
                self._complete_state(state, position, states)
            else:
                next_symbol = state.next_symbol()
                if next_symbol in self.grammar.terminals:
                    self._scan_state(state, position, states)
                else:
                    self._predict_state(state, position, states)

    def _predict_state(self, state: State, position: int, states: list[State]) -> None:
        """Handle the prediction step."""
        next_symbol = state.next_symbol()
        if not next_symbol:
            return

        if next_symbol in self.leo_memo[position] and self._is_right_recursive(state.rule):
            memoized_state = self.leo_memo[position][next_symbol]
            self.chart.add(memoized_state, position, states)
            return

        for rule in self.grammar.rules_by_lhs.get(next_symbol, []):
            new_state = State(rule, 0, position, comment=f"predicted from {state}")
            self.chart.add(new_state, position, states)

        if next_symbol in self.nullable_nonterminals:
            new_state = State(state.rule, state.dot + 1, state.start, comment=f"nullable {next_symbol}")
            self.chart.add(new_state, position, states)

    def _scan_state(self, state: State, position: int, states: list[State]) -> None:
        """Handle the scanning step."""
        next_symbol = state.next_symbol()
        if next_symbol is not None and position < len(self.tokens) and self.tokens[position] == next_symbol:
            new_state = State(
                state.rule, state.dot + 1, state.start, comment=f"scanned {self.tokens[position]}"
            )
            new_state.back_pointers = [*state.back_pointers, self.tokens[position]]
            self.chart.add(new_state, position + 1, states)

    def _complete_state(self, state: State, position: int, states: list[State]) -> None:
        """Handle the completion step."""
        previous_states = list(self.chart[state.start].values())
        for s in previous_states:
            if s.next_symbol() == state.rule.lhs:
                new_state = State(s.rule, s.dot + 1, s.start, comment=f"completed from {state} and {s}")
                new_state.back_pointers = [*s.back_pointers, state]
                self.chart.add(new_state, position, states)

        if self._is_right_recursive(state.rule) and state.rule.lhs not in self.leo_memo[position]:
            self.leo_memo[position][state.rule.lhs] = state

    def _build_parse_tree(self, state: State) -> tuple[str, list]:
        """Recursively build the parse tree from completed states."""
        if not state.back_pointers:
            return (state.rule.lhs, [])
        children = [self._build_parse_tree(bp) if isinstance(bp, State) else bp for bp in state.back_pointers]
        return (state.rule.lhs, children)


def print_tree(node: tuple[str, list] | str, indent: str = "", *, last: bool = True) -> None:
    """Recursively print the parse tree."""
    match node:
        case (label, children) if all(isinstance(c, str) for c in children):
            pointer = "└─" if last else "├─"
            print(f"{indent}{pointer}{label}: {", ".join(repr(c) for c in children)}")
        case (label, children):
            pointer = "└─" if last else "├─"
            print(f"{indent}{pointer}{label}")
            new_indent = indent + ("  " if last else "│ ")
            for i, child in enumerate(children):
                print_tree(child, new_indent, last=(i == len(children) - 1))
        case _:
            pointer = "└─" if last else "├─"
            print(f"{indent}{pointer}{node!r}")


def print_step(i: int, states: dict, tokens: list[str], *, fancy: bool = True) -> None:
    """Print the current step of the parser."""
    title = f"S({i}): {" ".join(repr(e) for e in tokens[:i])} • {" ".join(repr(e) for e in tokens[i:])}"
    if fancy:
        table = Table(title=title)
        table.add_column("Production")
        table.add_column("Origin")
        table.add_column("Comment")

        for state in states.values():
            table.add_row(repr(state), str(state.start), state.comment)
        console.print(table)
    else:
        console.print(title)
        for state in states.values():
            console.print(repr(state), str(state.start), state.comment)


def print_chart(chart: list[dict], tokens: list[str], *, fancy: bool = True) -> None:
    """Print the entire chart after parsing."""
    for i, states in enumerate(chart):
        print()
        print_step(i, states, tokens, fancy=fancy)


# Example usage:
if __name__ == "__main__":
    rules = [
        Rule("S", ["S", "+", "M"]),
        Rule("S", ["M"]),
        Rule("M", ["M", "*", "T"]),
        Rule("M", ["T"]),
        Rule("T", ["1"]),
        Rule("T", ["2"]),
        Rule("T", ["3"]),
        Rule("T", ["4"]),
    ]
    grammar = Grammar(rules, "S")

    tokens = ["2", "+", "3", "*", "4"]

    parser = EarleyParser(grammar, verbose=True)
    parse_tree = parser.parse(tokens)
    print_chart(parser.chart.chart, tokens, fancy=True)
    if parse_tree:
        print()
        print("Parse Tree")
        print_tree(parse_tree)
    else:
        print("Parsing failed.")
