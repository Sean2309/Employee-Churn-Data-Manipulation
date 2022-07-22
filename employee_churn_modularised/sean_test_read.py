import jazz_context
import jazz_context_show
import jazz_context_unshow
import jazz_antiformat
import briefology_ps_to_task_search as briefology_graph
import opera_util_common

def get_ctx():

    C = jazz_context.context
    antiformat = jazz_antiformat.main
    show = jazz_context_show.show
    unshow = jazz_context_unshow.unshow

    file = 'sean_test_config'
    c = opera_util_common.parse_ps_to_ctx(file)
    return c


