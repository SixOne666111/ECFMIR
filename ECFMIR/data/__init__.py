benchmarks = {
    'MIntRec':{
        'intent_labels': [
                    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask for help' 
        ],
        'binary_maps': {
                    'Complain': 'Emotion', 'Praise':'Emotion', 'Apologise': 'Emotion', 'Thank':'Emotion', 'Criticize': 'Emotion',
                    'Care': 'Emotion', 'Agree': 'Emotion', 'Taunt': 'Emotion', 'Flaunt': 'Emotion',
                    'Joke':'Emotion', 'Oppose': 'Emotion', 
                    'Inform':'Goal', 'Advise':'Goal', 'Arrange': 'Goal', 'Introduce': 'Goal', 'Leave':'Goal',
                    'Prevent':'Goal', 'Greet': 'Goal', 'Ask for help': 'Goal', 'Comfort': 'Goal'
        },
        'binary_intent_labels': ['Emotion', 'Goal'],
        'label_len': 4,
        'max_seq_lengths':{
            'text': 30,
            'video': 230,
            'audio': 480,
        },
        'feat_dims':{
            'text': 768,
            'video': 256,
            'audio': 768
        }
    },

    'MIntRec2': {
        'intent_labels': [
            'Introduce', 'Inform', 'Explain', 'Greet', 'Ask for help',
            'Thank', 'Confirm', 'Agree', 'Apologise', 'Arrange', 'Complain',
            'Advise', 'Acknowledge', 'Warn', 'Taunt', 'Criticize', 'Care',
            'Invite', 'Comfort', 'Praise', 'Flaunt', 'Emphasize', 'Leave',
            'Prevent', 'Oppose', 'Plan', 'Doubt', 'Joke',
            'Asking for opinions', 'Refuse', 'UNK'
        ],
        'label_len': 4,
        'max_seq_lengths': {
            'text': 76,
            'audio': 400,
            'video': 180
        },
        'feat_dims': {
            'text': 768,
            'audio': 768,
            'video': 256
        },
    },

    'MELD':{
        'intent_labels': [
                    'Greeting', 'Question', 'Answer', 'Statement Opinion', 'Statement Non Opinion',
                    'Apology', 'Command', 'Agreement', 'Disagreement',
                    'Acknowledge', 'Backchannel', 'Others'
        ],
        'label_maps': {
                    'g': 'Greeting', 'q': 'Question', 'ans': 'Answer', 'o': 'Statement Opinion', 's': 'Statement Non Opinion',
                    'ap': 'Apology', 'c': 'Command', 'ag': 'Agreement', 'dag': 'Disagreement',
                    'a': 'Acknowledge', 'b': 'Backchannel', 'oth': 'Others'
        },
        'label_len': 3,
        'max_seq_lengths':{
            'text': 70,
            'audio': 530,
            'video': 250
        },
        'feat_dims':{
            'text': 768,
            'audio': 768,
            'video': 1024
        }
    }
}

