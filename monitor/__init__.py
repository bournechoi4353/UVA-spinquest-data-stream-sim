"""The monitoring pipeline that consumes the stream.

    events (replay or the live reader)
        -> detectors.py   plain-code detectors, one ChannelMonitor per channel
        -> state.py       the state store: current conclusions per channel and feed
        -> summary.py     a short factual text summary of the state
        -> the LLM        phrases answers from the summary only (see training/ and bot/)

Detectors never see labels. Scoring against labels happens in scoring.py.
"""
