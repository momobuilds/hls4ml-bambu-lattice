import filecmp
import numpy as np
import os
import pytest
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

import hls4ml

test_root_path = Path(__file__).parent

# -----------------------------------------------------------------------------
# fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def hls_model_setup(test_case_id):
    """Fixture to create basic HLS model for Bambu backend"""
    proj_dir = test_root_path / test_case_id
    clean_proj_dir(proj_dir)

    model = Sequential()
    model.add(Dense(5, input_shape=(16,), name='fc1', activation='relu'))

    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(proj_dir),
        backend="Bambu",
    )

    yield hls_model

@pytest.fixture
def fake_completed_process():
    """Fixture that returns a fake completed subprocess result."""
    return SimpleNamespace(returncode=0, stdout='', stderr='')

@pytest.fixture
def fake_failed_process():
    """Fixture that returns a fake failed subprocess result."""
    return SimpleNamespace(returncode=1, stdout='', stderr='')

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def patch_subprocess(monkeypatch, completed):
    run = Mock(return_value=completed)
    monkeypatch.setattr("subprocess.run", run)
    return run


def patch_bambu_available(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/bambu")


def patch_bambu_unavailable(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)

def count_ext_files(directory, extension):
    """Counts how many files in the directory and all its
    subdirectories have files that end with extension"""
    return sum(1 for _ in Path(directory).rglob(f"*{extension}"))

def clean_proj_dir(proj_dir):
    if proj_dir.exists():
        shutil.rmtree(proj_dir)

# -----------------------------------------------------------------------------
# tests
# -----------------------------------------------------------------------------

ARGS = {
    "empty_str": {
        "input": "", 
        "expected": []
        },
    "none": {
        "input": None, 
        "expected": []
        },
    "empty_list": {
        "input": [], 
        "expected": []
        },
    "empty_tuple": {
        "input": (), 
        "expected": []
        },
    "string_with_single_quotes": {
        "input": "--extra-gcc-options='-emit-llvm -S'", 
        "expected": ["--extra-gcc-options=-emit-llvm -S"]
        },
    "string_with_double_quotes": {
        "input": '--extra-gcc-options="-emit-llvm -S"', 
        "expected": ["--extra-gcc-options=-emit-llvm -S"]
        },
    "list_args": {
        "input": ['--simulate', '--print-dot'], 
        "expected": ['--simulate', '--print-dot']
        },
    "tuple_args": {
        "input": ('--simulate', '--print-dot'), 
        "expected": ['--simulate', '--print-dot']
        },
}
@pytest.mark.parametrize("case", ARGS.values(), ids=ARGS.keys())
def test_args_normalization(case, hls_model_setup):
    """Test that user arguments are normalized and appended correctly to the default Bambu command."""
    args = case["input"]
    expected = case["expected"]

    model = hls_model_setup
    results = model.build(synth=False, args=args)

    # Defaults
    BASE_COMMAND = ['bambu', os.path.join('firmware', f'{model.config.get_project_name()}.cpp'),
                    f'--top-fname={model.config.get_project_name()}']
    
    # Current flags needed for HLS4ML and Bambu compatability
    # Likely to change as Bambu updates
    REQ_ARGS = ['-lm', 
                '-Ifirmware/ac_types',
                '--compiler=I386_CLANG16',
                '--generate-interface=INFER'
               ]

    # First check default command
    assert results["command"][:len(BASE_COMMAND)] == BASE_COMMAND

    # Then check that required AND user args are appended
    for token in (REQ_ARGS + expected):
        assert token in results["command"][len(BASE_COMMAND):]


def test_type_exceptions(hls_model_setup):
    """Test that invalid argument types to build() raise the correct exceptions."""
    model = hls_model_setup

    # args must be list, tuple, or string
    with pytest.raises(TypeError):
        model.build(args=0)

    # env must be mapping
    with pytest.raises(TypeError):
        model.build(env="not_a_dict")

    # run_kwargs must be mapping
    with pytest.raises(TypeError):
        model.build(run_kwargs="not_a_dict")

    # log_to_stdouot + stdout/stderr in run_kwargs
    with pytest.raises(ValueError):
        model.build(log_to_stdout=True, run_kwargs={'stdout': None})


def test_workflow_exceptions(hls_model_setup):
    """Test that invalid workflow combinations to build() raise the correct exceptions."""
    model = hls_model_setup

    # cosim
    with pytest.raises(ValueError):
        model.build(synth=False, cosim=True)

    # validation
    with pytest.raises(ValueError):
        model.build(csim=False, cosim=True, validation=True)
    with pytest.raises(ValueError):
        model.build(csim=True, cosim=False, validation=True)
    with pytest.raises(ValueError):
        model.build(csim=False, cosim=False, validation=True)

    # vsynth
    with pytest.raises(ValueError):
        model.build(synth=False, vsynth=True)
    with pytest.raises(ValueError):
        model.build(cosim=False, vsynth=True)


def test_subprocess_failure_exceptions(hls_model_setup, fake_failed_process, monkeypatch):
    """Test that failed (return code != 0) subprocesses raise RuntimeError"""
    patch_bambu_available(monkeypatch)
    patch_subprocess(monkeypatch, fake_failed_process)

    model = hls_model_setup
    with pytest.raises(RuntimeError):
        model.build(csim=True, synth=False)
    with pytest.raises(RuntimeError):
        model.build(csim=False, synth=True)


@pytest.mark.parametrize(
    "env, expected_present, expected_missing",
    [
        ({"FOO": "1"}, ["FOO"], []),
        ({"FOO": None}, [], ["FOO"]),
    ],
)
def test_env_handling(
    monkeypatch,
    hls_model_setup,
    fake_completed_process,
    env,
    expected_present,
    expected_missing,
):
    """Test that the environment passed to subprocess.run is correctly modified according to `env`."""
    patch_bambu_available(monkeypatch)
    run = patch_subprocess(monkeypatch, fake_completed_process)

    model = hls_model_setup
    model.build(env=env)

    _, kwargs = run.call_args
    run_env = kwargs["env"]

    # added env variables
    for key in expected_present:
        assert run_env[key] == env[key]

    # deleted env variables
    for key in expected_missing:
        assert key not in run_env

    # make sure rest of environment unaffected
    for key, value in os.environ.items():
        if key in env and env[key] is None:
            assert key not in run_env
        elif key in env:
            assert run_env[key] == env[key]
        else:
            assert run_env[key] == value


def test_bambu_missing_raises(hls_model_setup, monkeypatch):
    """Test that build() raises an EnvironmentError if Bambu is not found on PATH."""
    patch_bambu_unavailable(monkeypatch)

    model = hls_model_setup

    with pytest.raises(EnvironmentError):
        model.build()


def test_csim(tmp_path, test_case_id):
    """Test that csim produces similar results to the
    CPP implementation of the HLS4ML model.
    """
    csim_proj_dir = test_root_path / test_case_id

    model = Sequential()
    model.add(Dense(5, input_shape=(16,), name='fc1', activation='relu'))

    # Create CPP bridge predictions
    clean_proj_dir(csim_proj_dir)
    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(csim_proj_dir),
        backend="Bambu"
    )
    hls_model.compile()

    X = np.random.rand(8, 16).astype(np.float32)
    Y = hls_model.predict(X)
    np.save(tmp_path / 'input.npy', X)
    np.save(tmp_path / 'output.npy', Y)

    # Create Bambu csim predictions
    clean_proj_dir(csim_proj_dir)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(csim_proj_dir),
        backend="Bambu",
        input_data_tb=str(tmp_path / 'input.npy'),
        output_data_tb=str(tmp_path / 'output.npy')
    )
    result = hls_model.build(csim=True)

    Y_bridge = np.loadtxt(csim_proj_dir / 'tb_data' / 'tb_output_predictions.dat')
    Y_csim = result['predictions_csim']

    assert np.allclose(Y_bridge, Y_csim, rtol=0, atol=1e-4)


def test_cosim(tmp_path, test_case_id):
    """Test that cosim produces similar results to the
    CPP implementation of the HLS4ML model.
    """
    cosim_proj_dir = test_root_path / test_case_id

    model = Sequential()
    model.add(Dense(5, input_shape=(16,), name='fc1', activation='relu'))

    # Create CPP bridge predictions
    clean_proj_dir(cosim_proj_dir)
    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(cosim_proj_dir),
        backend="Bambu"
    )
    hls_model.compile()

    X = np.random.rand(8, 16).astype(np.float32)
    Y = hls_model.predict(X)
    np.save(tmp_path / 'input.npy', X)
    np.save(tmp_path / 'output.npy', Y)

    # Create Bambu cosim predictions
    clean_proj_dir(cosim_proj_dir)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(cosim_proj_dir),
        backend="Bambu",
        input_data_tb=str(tmp_path / 'input.npy'),
        output_data_tb=str(tmp_path / 'output.npy')
    )
    result = hls_model.build(synth=True, cosim=True)

    Y_bridge = np.loadtxt(cosim_proj_dir / 'tb_data' / 'tb_output_predictions.dat')
    Y_cosim = result['predictions_cosim']

    assert np.allclose(Y_bridge, Y_cosim, rtol=0, atol=1e-4)


def test_validation_output(hls_model_setup, test_case_id):
    """Test that validation produces the two testbench output files and validates
    them correctly. This also tests csim and cosim, because validation does not run/
    is not valid without running both.
    """
    proj_dir = test_root_path / test_case_id
    model = hls_model_setup
    result = model.build(csim=True, synth=True, cosim=True, validation=True)

    # csim
    csim_path = proj_dir / 'tb_data' / 'csim_results.log'
    assert csim_path.exists()

    # cosim
    cosim_path = proj_dir / 'tb_data' / 'rtl_cosim_results.log'
    assert cosim_path.exists()

    # validation
    if filecmp.cmp(csim_path, cosim_path, shallow=False):
        assert result['valid'] == True
    else:
        assert result['valid'] == False


def test_vsynth_output(test_case_id):
    """Test that a successful vsynth run produces the desired reports.
    Uses 7-Series Artix part "xc7a100tcsg324-1" to synthesize in Vivado.
    """
    vsynth_proj_dir = test_root_path / test_case_id

    # Create synthesizable HLS4ML model
    model = Sequential()
    model.add(Dense(5, input_shape=(16,), name='fc1', activation='relu'))

    clean_proj_dir(vsynth_proj_dir)
    config = hls4ml.utils.config_from_keras_model(model, granularity='model')
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=str(vsynth_proj_dir),
        backend="Bambu",
        clock_period=10,
        part_name='xc7a100tcsg324-1'
    )
    hls_model.build(synth=True, cosim=True, vsynth=True)

    # Ensure we get bambu results file
    assert sum(1 for _ in vsynth_proj_dir.rglob("bambu_results_*.xml")) >= 1

    # Ensure we get expected reports
    num_reports = count_ext_files(vsynth_proj_dir / 'HLS_output', '.rpt')
    assert num_reports >= 15

