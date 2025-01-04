from streamlit.testing.v1 import AppTest

def test_draft_defaults():
    """Make sure the draft mode is set up correctly in terms of default parameters"""
    at = AppTest.from_file("../app.py").run(timeout = 300)

    #make sure parameters are initialized correctly
    assert not at.session_state.run_h_score
    assert at.session_state.mode == 'Draft Mode'
    assert at.session_state.params is not None

    #make sure that the defaults for the options page are set up correctly 
    for option_name in ['n_drafters','n_picks','psi','nu','omega','gamma','alpha','beta','n_iterations']:
        assert at.number_input(option_name).value == at.session_state.params['options'][option_name]['default']
        assert at.number_input(option_name).min == at.session_state.params['options'][option_name]['min']
        assert at.number_input(option_name).max == at.session_state.params['options'][option_name]['max']

    #check what tabs are available? is that possible?