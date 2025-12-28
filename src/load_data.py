import wfdb

def load_ecg_record(record_id):
    """Sinyal ve anotasyonları yükle."""
    record = wfdb.rdrecord(record_id, pn_dir='mitdb')
    annotation = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
    return record.p_signal[:, 0], annotation.sample, record.fs