class DataIngestionConfig:
    def __init__(self, source_file, intents_dict, threshold):
        self.source_file = source_file
        self.intents_dict = intents_dict
        self.threshold = threshold


class BaseModelPreparationConfig:
    def __init__(self, base_model_name, intents):
        self.base_model_name = base_model_name
        self.intents = intents

    def to_dict(self):
        return {
            'base_model_name': self.base_model_name,
            'intents': self.intents
        }


class CallbacksPreparationConfig:
    def __init__(self, tensorboard_log_dir, checkpoint_filepath):
        self.tensorboard_log_dir = tensorboard_log_dir
        self.checkpoint_filepath = checkpoint_filepath

    def to_dict(self):
        return {
            'tensorboard_log_dir': self.tensorboard_log_dir,
            'checkpoint_filepath': self.checkpoint_filepath,
            # include other properties as needed...
        }



class TrainingConfig:
    def __init__(self, epochs, base_model_name, intents):
        self.epochs = epochs
        self.base_model_name = base_model_name
        self.intents = intents

    def to_dict(self):
        return {
            'epochs': self.epochs,
            'base_model_name': self.base_model_name,
            'intents': self.intents
        }
class ConfigurationManager:
    def __init__(self):
        pass

    

    def get_data_ingestion_config(self):
        # Replace with your actual values
        source_file = 'C:/work/MLOPS/Chicken-Disease-Classification-Projects-main/Chicken-Disease-Classification-Projects-main/src/cnnClassifier/config/blogtext.csv'
        intents_dict = {
    "informative_dict" : ["information", "facts", "detail", "data", "knowledge", "clarify", "explain", "report", "reveal", "show", "statistics", "study", "update", "news", "research", "analysis", "discover", "learn", "reference", "insight", "summary", "overview", "guidance", "resource", "announcement", "result", "finding", "document", "outline", "intel", "source", "evidence", "proof", "understand", "notification", "illuminate", "dissemination", "briefing", "demonstration", "exposition", "discovery", "observation", "deliver", "awareness", "breaking", "fresh", "recent", "verify", "comprehend", "grasp", "master", "glean", "survey", "uncover", "realize", "establish", "conclude", "digest", "assimilate", "catch", "perceive", "discern", "deduce", "note", "learned", "erudite", "scholarly", "informed", "enlightened", "absorb", "knowledgeable", "educated", "lettered", "well-read", "bookish", "literate", "savvy", "brainy", "cerebral", "cultured", "sophisticated", "grounded", "apprehend", "fathom", "sense", "detect", "identify", "recognition", "notice", "spot", "locate", "find", "disclose", "unveil", "expose", "unearth", "excavate", "dredge", "quarry", "mine", "bring to light"],

    "opinion_dict" : ["think", "believe", "view", "feel", "consider", "judgment", "assessment", "perspective", "stance", "standpoint", "angle", "slant", "vantage point", "point of view", "outlook", "position", "attitude", "opine", "reckon", "deem", "conjecture", "speculate", "guess", "hypothesize", "theorize", "postulate", "surmise", "presume", "assume", "infer", "deduce", "suspect", "guesstimate", "estimate", "evaluate", "interpret", "analyze", "appraise", "value", "rate", "rank", "grade", "weigh", "examine", "scrutinize", "study", "probe", "investigate", "explore", "review", "inspect", "peruse", "scan", "vet", "audit", "check", "test", "question", "query", "inquire", "probe", "doubt", "challenge", "skeptic", "cynic", "distrust", "mistrust", "disbelieve", "unconvinced", "incredulous", "discrediting", "refute", "dispute", "debunk", "denounce", "criticize", "condemn", "decry", "blast", "slam", "knock", "pan", "lambast", "roast", "slate", "deride", "mock", "scoff", "ridicule", "jibe", "jeer", "sneer", "scorn", "satirize", "lampoon", "caricature", "parody", "tease", "ridicule", "taunt"],

    "promotional_dict" : ["promote", "sell", "market", "advertise", "offer", "endorse", "sponsor", "boost", "push", "plug", "hype", "publicize", "publish", "broadcast", "announce", "proclaim", "expose", "reveal", "display", "showcase", "exhibit", "present", "feature", "highlight", "spotlight", "flaunt", "parade", "show off", "brag", "boast", "vaunt", "trumpet", "tout", "ballyhoo", "blazon", "splash", "puff", "flog", "hawk", "peddle", "punt", "pitch", "propaganda", "persuade", "convince", "sway", "influence", "coax", "cajole", "charm", "win over", "entice", "tempt", "lure", "draw", "attract", "catch", "captivate", "enchant", "allure", "fascinate", "bewitch", "seduce", "woo", "court", "bait", "tease", "titillate", "provoke", "stir", "excite", "stimulate", "arouse", "fire up", "whip up", "kindle", "ignite", "spark", "fan", "fuel", "stoke", "heat", "blaze", "burn", "glow", "flare", "flame", "inflame"],

    "entertaining_dict" : ["fun", "humor", "joke", "laugh", "entertain", "amuse", "enjoy", "play", "game", "leisure", "recreation", "diversion", "pastime", "hobby", "amusement", "pleasure", "delight", "joy", "gaiety", "glee", "mirth", "cheer", "cheerfulness", "cheeriness", "merriment", "merriness", "happiness", "gladness", "joviality", "jollity", "jolliness", "jocularity", "jocundity", "festivity", "celebration", "party", "feast", "banquet", "gathering", "social", "mixer", "gettogether", "soiree", "gala", "ball", "bash", "shindig", "hootenanny", "rave", "fete", "fiesta", "convivial", "merry", "jolly", "joyful", "joyous", "blithe", "light-hearted", "carefree", "free-spirited", "spirited", "lively", "vivacious", "animated", "energetic", "effervescent", "bubbly", "exuberant", "ebullient", "zestful", "zealous", "enthusiastic", "eager", "keen", "fervent", "passionate", "ardent", "avid", "fired up", "pumped up", "excited", "thrilled", "electrified", "stimulated", "titillated"],
    "educational_dict" : ["teach", "educate", "instruct", "school", "train", "tutor", "coach", "mentor", "guide", "lead", "show", "direct", "point", "indicate", "demonstrate", "model", "exemplify", "illustrate", "enlighten", "illuminate", "shed light", "clarify", "explain", "elucidate", "define", "describe", "expound", "detail", "unpack", "break down", "decipher", "decode", "translate", "interpret", "read", "spell out", "lay out", "put across", "get across", "communicate", "convey", "impart", "pass on", "transmit", "relay", "send", "deliver", "give", "offer", "provide", "supply", "furnish", "equip", "arm", "prep", "prime", "ready", "condition", "drill", "exercise", "practice", "workout", "rehearse", "run through", "walk through", "go over", "review", "revise", "correct", "edit", "check", "inspect", "examine", "scrutinize", "study", "probe", "explore", "research", "investigate", "inquire", "query", "question", "ask", "probe", "pry", "nose around", "snoop", "dig", "delve", "burrow", "tunnel", "mine", "quarry", "excavate"],

    "inspirational_dict" : ["inspire", "motivate", "encourage", "influence", "persuade", "convince", "sway", "move", "stir", "rouse", "excite", "spark", "ignite", "kindle", "fire", "fuel", "stimulate", "provoke", "incite", "galvanize", "energize", "activate", "drive", "push", "spur", "prompt", "propel", "impel", "urge", "egg on", "goad", "prod", "poke", "prick", "jab", "nudge", "press", "pressure", "coax", "cajole", "charm", "entice", "tempt", "lure", "draw", "attract", "catch", "engage", "captivate", "fascinate", "enchant", "beguile", "mesmerize", "hypnotize", "spellbind", "allure", "woo", "seduce", "win over", "touch", "affect", "impact", "hit", "strike", "punch", "slap", "smack", "knock", "whack", "thwack", "wallop", "pound", "hammer", "bang", "clout", "clobber", "belt", "bash", "biff", "bop", "sock", "slug", "smite", "swat", "thump"],

}

        threshold = 3
        return DataIngestionConfig(source_file, intents_dict, threshold)
    def get_training_config(self):
        # Replace with your actual values
        source_file = 'C:/work/MLOPS/Chicken-Disease-Classification-Projects-main/Chicken-Disease-Classification-Projects-main/src/cnnClassifier/config/blogtext.csv'
        intents_dict = {
    "informative_dict" : ["information", "facts", "detail", "data", "knowledge", "clarify", "explain", "report", "reveal", "show", "statistics", "study", "update", "news", "research", "analysis", "discover", "learn", "reference", "insight", "summary", "overview", "guidance", "resource", "announcement", "result", "finding", "document", "outline", "intel", "source", "evidence", "proof", "understand", "notification", "illuminate", "dissemination", "briefing", "demonstration", "exposition", "discovery", "observation", "deliver", "awareness", "breaking", "fresh", "recent", "verify", "comprehend", "grasp", "master", "glean", "survey", "uncover", "realize", "establish", "conclude", "digest", "assimilate", "catch", "perceive", "discern", "deduce", "note", "learned", "erudite", "scholarly", "informed", "enlightened", "absorb", "knowledgeable", "educated", "lettered", "well-read", "bookish", "literate", "savvy", "brainy", "cerebral", "cultured", "sophisticated", "grounded", "apprehend", "fathom", "sense", "detect", "identify", "recognition", "notice", "spot", "locate", "find", "disclose", "unveil", "expose", "unearth", "excavate", "dredge", "quarry", "mine", "bring to light"],

    "opinion_dict" : ["think", "believe", "view", "feel", "consider", "judgment", "assessment", "perspective", "stance", "standpoint", "angle", "slant", "vantage point", "point of view", "outlook", "position", "attitude", "opine", "reckon", "deem", "conjecture", "speculate", "guess", "hypothesize", "theorize", "postulate", "surmise", "presume", "assume", "infer", "deduce", "suspect", "guesstimate", "estimate", "evaluate", "interpret", "analyze", "appraise", "value", "rate", "rank", "grade", "weigh", "examine", "scrutinize", "study", "probe", "investigate", "explore", "review", "inspect", "peruse", "scan", "vet", "audit", "check", "test", "question", "query", "inquire", "probe", "doubt", "challenge", "skeptic", "cynic", "distrust", "mistrust", "disbelieve", "unconvinced", "incredulous", "discrediting", "refute", "dispute", "debunk", "denounce", "criticize", "condemn", "decry", "blast", "slam", "knock", "pan", "lambast", "roast", "slate", "deride", "mock", "scoff", "ridicule", "jibe", "jeer", "sneer", "scorn", "satirize", "lampoon", "caricature", "parody", "tease", "ridicule", "taunt"],

    "promotional_dict" : ["promote", "sell", "market", "advertise", "offer", "endorse", "sponsor", "boost", "push", "plug", "hype", "publicize", "publish", "broadcast", "announce", "proclaim", "expose", "reveal", "display", "showcase", "exhibit", "present", "feature", "highlight", "spotlight", "flaunt", "parade", "show off", "brag", "boast", "vaunt", "trumpet", "tout", "ballyhoo", "blazon", "splash", "puff", "flog", "hawk", "peddle", "punt", "pitch", "propaganda", "persuade", "convince", "sway", "influence", "coax", "cajole", "charm", "win over", "entice", "tempt", "lure", "draw", "attract", "catch", "captivate", "enchant", "allure", "fascinate", "bewitch", "seduce", "woo", "court", "bait", "tease", "titillate", "provoke", "stir", "excite", "stimulate", "arouse", "fire up", "whip up", "kindle", "ignite", "spark", "fan", "fuel", "stoke", "heat", "blaze", "burn", "glow", "flare", "flame", "inflame"],

    "entertaining_dict" : ["fun", "humor", "joke", "laugh", "entertain", "amuse", "enjoy", "play", "game", "leisure", "recreation", "diversion", "pastime", "hobby", "amusement", "pleasure", "delight", "joy", "gaiety", "glee", "mirth", "cheer", "cheerfulness", "cheeriness", "merriment", "merriness", "happiness", "gladness", "joviality", "jollity", "jolliness", "jocularity", "jocundity", "festivity", "celebration", "party", "feast", "banquet", "gathering", "social", "mixer", "gettogether", "soiree", "gala", "ball", "bash", "shindig", "hootenanny", "rave", "fete", "fiesta", "convivial", "merry", "jolly", "joyful", "joyous", "blithe", "light-hearted", "carefree", "free-spirited", "spirited", "lively", "vivacious", "animated", "energetic", "effervescent", "bubbly", "exuberant", "ebullient", "zestful", "zealous", "enthusiastic", "eager", "keen", "fervent", "passionate", "ardent", "avid", "fired up", "pumped up", "excited", "thrilled", "electrified", "stimulated", "titillated"],
    "educational_dict" : ["teach", "educate", "instruct", "school", "train", "tutor", "coach", "mentor", "guide", "lead", "show", "direct", "point", "indicate", "demonstrate", "model", "exemplify", "illustrate", "enlighten", "illuminate", "shed light", "clarify", "explain", "elucidate", "define", "describe", "expound", "detail", "unpack", "break down", "decipher", "decode", "translate", "interpret", "read", "spell out", "lay out", "put across", "get across", "communicate", "convey", "impart", "pass on", "transmit", "relay", "send", "deliver", "give", "offer", "provide", "supply", "furnish", "equip", "arm", "prep", "prime", "ready", "condition", "drill", "exercise", "practice", "workout", "rehearse", "run through", "walk through", "go over", "review", "revise", "correct", "edit", "check", "inspect", "examine", "scrutinize", "study", "probe", "explore", "research", "investigate", "inquire", "query", "question", "ask", "probe", "pry", "nose around", "snoop", "dig", "delve", "burrow", "tunnel", "mine", "quarry", "excavate"],

    "inspirational_dict" : ["inspire", "motivate", "encourage", "influence", "persuade", "convince", "sway", "move", "stir", "rouse", "excite", "spark", "ignite", "kindle", "fire", "fuel", "stimulate", "provoke", "incite", "galvanize", "energize", "activate", "drive", "push", "spur", "prompt", "propel", "impel", "urge", "egg on", "goad", "prod", "poke", "prick", "jab", "nudge", "press", "pressure", "coax", "cajole", "charm", "entice", "tempt", "lure", "draw", "attract", "catch", "engage", "captivate", "fascinate", "enchant", "beguile", "mesmerize", "hypnotize", "spellbind", "allure", "woo", "seduce", "win over", "touch", "affect", "impact", "hit", "strike", "punch", "slap", "smack", "knock", "whack", "thwack", "wallop", "pound", "hammer", "bang", "clout", "clobber", "belt", "bash", "biff", "bop", "sock", "slug", "smite", "swat", "thump"],

}

        threshold = 3
        return TrainingConfig(source_file, intents_dict, threshold)

    def get_base_model_preparation_config(self):
        # Replace with your actual values
        base_model_name = 'bert-base-uncased'
        intents = list(self.get_data_ingestion_config().intents_dict.keys())
        return BaseModelPreparationConfig(base_model_name, intents)

    def get_callbacks_preparation_config(self):
        # Replace with your actual values
        tensorboard_log_dir = './logs'
        checkpoint_filepath = './checkpoint'
        return CallbacksPreparationConfig(tensorboard_log_dir, checkpoint_filepath)

    def get_training_config(self):
        # Replace with your actual values
        epochs = 10
        base_model_name = 'bert-base-uncased'
        intents = ['intent1', 'intent2', 'intent3']
        return TrainingConfig(epochs, base_model_name, intents)
