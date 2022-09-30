import dsl

DSL_DICT = {
			('list', 'list') : [],
			('list', 'atom') : [dsl.LogicAndFunction, dsl.LogicOrFunction, dsl.AffineFunction],
            ('atom', 'atom') : [dsl.SimpleITE]
}

CUSTOM_EDGE_COSTS = {
	('list', 'list') : {},
	('list', 'atom') : {},
	('atom', 'atom') : {}
}