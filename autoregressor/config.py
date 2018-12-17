from pathlib import Path
REPO_ROOT = Path(__file__).parent
# Here are stored models and checpoint files created during automatic tests.
# Such files are not cleaned up automatically after test but removed before test (if exist) 
# So they can be inspected after test is executed.
TEST_TMP_DIR = REPO_ROOT/"tmp_test_data"
VOCABULARIES_BASE_DIR = REPO_ROOT/"vocabularies"