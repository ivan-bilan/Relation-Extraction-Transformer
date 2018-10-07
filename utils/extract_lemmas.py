import re


def extract_lemmas(spacy_nlp, tokens, i):
    # TODO: do more experiments with lemmas

    init_tokens = tokens
    # if lemma
    # use lemmas instead of raw text
    tokens_1_len = len(tokens)
    # print(len(tokens))
    # print(tokens)

    tokens = u' '.join(tokens)

    # TODO: find a way to get rid of the regex plugs and integrate lemmas into batching directly
    # do this twice
    tokens = re.sub(r"(\w),?\.?-(\w)", "\g<1>_\g<2>", tokens)
    tokens = re.sub(r"(\w),(\w)", "\g<1>_\g<2>", tokens)

    tokens = re.sub(r"(\w)-+(\w)", "\g<1>_\g<2>", tokens)
    # tokens = re.sub(r"(\w)/(\w)", "\g<1>_\g<2>", tokens)

    tokens = re.sub(r"(\w)/(\w)/?(\w){,3}?/?(\w){,3}?", "\g<1>_\g<2>", tokens)

    tokens = re.sub(r"(\w)\.+([\w@])", "\g<1>_\g<2>", tokens)
    tokens = re.sub(r" '(\w)", " \g<1>", tokens)
    tokens = re.sub(r" '(\d)", " \g<1>", tokens)  # ?
    tokens = re.sub(r" \+(\d)", " \g<1>", tokens)
    tokens = re.sub(r" ,(\w)", " \g<1>", tokens)
    tokens = re.sub(r" ,(\d)", "\g<1>", tokens)
    # tokens = re.sub(r" :(\w)", " \g<1>", tokens)
    tokens = re.sub(r" [:#]([\d\w-])", " \g<1>", tokens)
    tokens = re.sub(r"^[:#]([\d\w-])", "\g<1>", tokens)

    tokens = re.sub(r"(\w)[:!?=](\w)", "\g<1>_\g<2>", tokens)
    tokens = re.sub(r"(\w)[:!?=]([A-Z])", "\g<1>_\g<2>", tokens)
    # tokens = re.sub(r"(\w)=(\w)", "\g<1>_\g<2>", tokens)

    tokens = re.sub(r" <(\w)", " \g<1>", tokens)
    tokens = re.sub(r"([\w\d])[>!?\]] ?", "\g<1> ", tokens)

    tokens = re.sub(r"(\w)&(\w)", "\g<1>_\g<2>", tokens)
    tokens = re.sub(r"([\w\d])& ", "\g<1> ", tokens)

    tokens = re.sub(r"(\w)\.", "\g<1>", tokens)
    tokens = re.sub(r"(\w)\* ", "\g<1> ", tokens)
    tokens = re.sub(r"(\w)'", "\g<1>", tokens)
    tokens = re.sub(r"(\w): ", "\g<1> ", tokens)
    tokens = re.sub(r"([\w\.]); ", "\g<1> ", tokens)
    tokens = re.sub(r"(\w)_ ", "\g<1> ", tokens)

    # ;P
    tokens = re.sub(r" ;([\d\w-])", " \g<1>", tokens)

    # normalize thousands
    tokens = re.sub(r"(\d+)K ", "\g<1>.000 ", tokens)
    tokens = re.sub(r"(\d+)[A-Za-z][A-Za-z]? ", "\g<1> ", tokens)
    tokens = re.sub(r"(\d+)[A-Za-z][A-Za-z]?$", "\g<1> ", tokens)
    tokens = re.sub(r"(\d+)m+ ", "\g<1> ", tokens)
    tokens = re.sub(r"(\d+)pm ", "\g<1> ", tokens)

    # trickery TODO: fix this!
    tokens = re.sub(r" [Ww]ed\.? ", " wedding ", tokens)
    tokens = re.sub(r" (couldnt|wouldnt) ", " would ", tokens)
    tokens = re.sub(r" wont ", " will ", tokens)
    tokens = re.sub(r" cant ", " can ", tokens)
    tokens = re.sub(r" didnt ", " did ", tokens)
    tokens = re.sub(r" thats ", " that ", tokens)
    tokens = re.sub(r"^thats ", "that ", tokens)
    tokens = re.sub(r" shes ", " she ", tokens)
    tokens = re.sub(r" hes ", " he ", tokens)
    tokens = re.sub(r" whats ", " what ", tokens)
    tokens = re.sub(r" wasnt ", " was ", tokens)
    tokens = re.sub(r" whos ", " who ", tokens)
    tokens = re.sub(r" shouldnt ", " should ", tokens)
    tokens = re.sub(r" theres ", " there ", tokens)
    tokens = re.sub(r" isnt ", " is ", tokens)
    tokens = re.sub(r" werent ", " were ", tokens)

    # TODO: ask about this on stackoverflow?
    tokens = re.sub(r" dont ", " do ", tokens)
    tokens = re.sub(r" doesnt ", " does ", tokens)

    tokens = re.sub(r"Cant ", "Can ", tokens)
    tokens = re.sub(r"Hes ", "He ", tokens)
    tokens = re.sub(r"Thats ", "That ", tokens)

    tokens = re.sub(r" Hed ", " He ", tokens)
    tokens = re.sub(r" [Ii]m ", " I ", tokens)
    tokens = re.sub(r"^[Ii]m ", "I ", tokens)
    # tokens = re.sub(r'[\?\!]+', '.', tokens)

    tokens = re.sub(r'([\!\?\*\_\=\.\#\']){1,}', '\g<1>', tokens)
    tokens = re.sub(r"(\w)\. ", "\g<1> ", tokens)
    tokens = re.sub(r"(\w)\# ", "\g<1> ", tokens)
    tokens = re.sub(r"(\w)=(\w)", "\g<1>_\g<2>", tokens)

    # TODO: normalize URLs amd emails?
    # tokens = re.sub(r'\?{1,}', '?', tokens)

    tokens = spacy_nlp(tokens)

    # TODO: leave pronouns back in
    # TODO: make it lower too
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    # [token.lemma_ for token in tokens]

    if tokens_1_len != len(tokens):

        print("Current sentence index:", i)
        print(tokens_1_len, len(tokens))
        print(init_tokens)
        print(tokens)

        for i, element in enumerate(init_tokens):
            if init_tokens[i] != tokens[i]:
                print("token:", init_tokens[i])
                print("posL", i)

    # TODO: if assertion fails, fall back to the original sentence!!!
    assert tokens_1_len == len(tokens)

    return tokens
