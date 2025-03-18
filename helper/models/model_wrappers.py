import torch.nn as nn
import torch
from tqdm import tqdm
import quba_constants as wc
import torch.nn.functional as F
from helper.imagenet import imagenet_templates as openai_imagenet_template
from abc import abstractmethod
import os

class AbstractModel(nn.Module):
    def __init__(self, model, model_name, transform):
        """
        An abstract wrapper for PyTorch models implementing functions required for evaluation.
        Args:
            model: PyTorch neural network model
        """
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.transform = transform

    def __call__(self, input):
        return self.model(input)

    @abstractmethod
    def forward(self, input):
        return self.model
    
    def to(self, device):
        self.model = self.model.to(device)

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

class StandardModel(AbstractModel):
    """
    A wrapper for standard PyTorch models (e.g. ResNet, VGG, AlexNet, ...).
    Args:
        model: PyTorch neural network model
    """

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

class DINOModel(StandardModel):
    """
    A wrapper for DINO
    """
    
    def __init__(self, model, head, model_name, transform):
        super().__init__(model, model_name, transform)
        self.head = head


    def __call__(self, input):
        n = 1 if "ViT-b-16-DINO-LP" == self.model_name else 4
        avgpool = True  if "ViT-b-16-DINO-LP" == self.model_name else False       
        intermediate_output = self.model.get_intermediate_layers(input, n)
        output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
        if avgpool:
            output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
            output = output.reshape(output.shape[0], -1)
        
        output = self.head(output)
        return output

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def to(self, device):
        self.model = self.model.to(device)
        self.head = self.head.to(device)

import clip
import helper.models.mobileclip as mobileclip
imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]
class ClipModel(StandardModel):
    """
    A wrapper for CLIP
    """
    def __init__(self, model_name, device):
        self.device = device
        model = None
        preprocess = None
     
        if "clip-resnet50" == model_name:
            model, preprocess = clip.load("RN50", device)
        elif "clip-vit-b-16" == model_name:
            model, preprocess = clip.load("ViT-B/16", device)
        elif "clip-resnet101" == model_name:
            model, preprocess = clip.load("RN101", device)
        elif "clip-vit-b-32" == model_name:
            model, preprocess = clip.load("ViT-B/32", device)
        elif "clip-vit-l-14" == model_name:
            model, preprocess = clip.load("ViT-L/14", device)
            
         
        model = model.to(device)
        super().__init__(model, model_name, transform=preprocess)

        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(imagenet_classes):
                texts = [template(class_name) for template in openai_imagenet_template]  # format with class
                texts = clip.tokenize(texts).to(device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            self._text_features = torch.stack(zeroshot_weights, dim=1).to(device)

    def __call__(self, image_input):
        image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ self._text_features).softmax(dim=-1)
        values = similarity
        return values

class MobileCLIPModel(StandardModel):
    """
    A wrapper for MobileCLIP
    """
    def __init__(self, model_name, device):
        self.device = device
        model = None
        preprocess = None
        tokenizer = None
        if model_name == "mobileclip-s0":
            download_file(url="https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s0.pt", new_name='mobileclip_s0.pt')
            model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=wc.MOBILECLIP_ROOT + 'mobileclip_s0.pt')
            tokenizer = mobileclip.get_tokenizer('mobileclip_s0').to(device)
        elif model_name == "mobileclip-s1":
            download_file(url="https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s1.pt", new_name='mobileclip_s1.pt')
            model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s1', pretrained=wc.MOBILECLIP_ROOT + 'mobileclip_s1.pt')
            tokenizer = mobileclip.get_tokenizer('mobileclip_s1').to(device)
        elif model_name == "mobileclip-s2":
            download_file(url="https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_s2.pt", new_name='mobileclip_s2.pt')
            model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s2', pretrained=wc.MOBILECLIP_ROOT + 'mobileclip_s2.pt')
            tokenizer = mobileclip.get_tokenizer('mobileclip_s2').to(device)
        elif model_name == "mobileclip-b":
            download_file(url="https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_b.pt", new_name='mobileclip_b.pt')
            model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained=wc.MOBILECLIP_ROOT + 'mobileclip_b.pt')
            tokenizer = mobileclip.get_tokenizer('mobileclip_b').to(device)
        elif model_name == "mobileclip-blt":
            download_file(url="https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt", new_name='mobileclip_blt.pt')
            model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_b', pretrained=wc.MOBILECLIP_ROOT + 'mobileclip_blt.pt')
            tokenizer = mobileclip.get_tokenizer('mobileclip_b').to(device)
        model.to(device)
        super().__init__(model, model_name, transform=preprocess)

        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(imagenet_classes):
                texts = [template(class_name) for template in openai_imagenet_template]  # format with class
                texts = tokenizer(texts).to(device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            self._text_features = torch.stack(zeroshot_weights, dim=1).to(device)

    def __call__(self, image_input):
        image_features = None
        if isinstance(self.model, torch.nn.DataParallel):
            image_features = self.model.module.encode_image(image_input)
        else:
            image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ self._text_features).softmax(dim=-1)
        values = similarity
        return values

class SigLIPModel(StandardModel):
    def __init__(self, model_name, model, tokenizer, preprocess, device):
        self.device = device

        super().__init__(model, model_name, transform=preprocess)
        
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(imagenet_classes):
                texts = [template(class_name) for template in openai_imagenet_template]  # format with class
                texts = tokenizer(texts, context_length=model.context_length).to(device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)

            self._text_features = torch.stack(zeroshot_weights, dim=1).to(device)


    def __call__(self, image_input):
        image_features = self.model.encode_image(image_input).to(self.device)
        image_features = F.normalize(image_features, dim=-1).to(self.device)
        text_probs = torch.sigmoid(image_features @ self._text_features * self.model.logit_scale.exp() + self.model.logit_bias).to(self.device)
        return text_probs

from open_clip import create_model_from_pretrained, get_tokenizer
class SigLIP2Model(StandardModel):

    def __init__(self, model_name, device):
        self.device = device

        model, preprocess = create_model_from_pretrained('hf-hub:timm/' + wc.SIGLIP2_MODELS[model_name])
        tokenizer = get_tokenizer('hf-hub:timm/' + wc.SIGLIP2_MODELS[model_name])

        super().__init__(model, model_name, transform=preprocess)
        
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(imagenet_classes):
                texts = [template(class_name) for template in openai_imagenet_template]  # format with class
                texts = tokenizer(texts, context_length=model.context_length).to(device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)

            self._text_features = torch.stack(zeroshot_weights, dim=1).to(device)


    def __call__(self, image_input):
        image_features = self.model.encode_image(image_input).to(self.device)
        image_features = F.normalize(image_features, dim=-1).to(self.device)
        text_probs = torch.sigmoid(image_features @ self._text_features * self.model.logit_scale.exp() + self.model.logit_bias).to(self.device)
        return text_probs

import open_clip
class OpenCLIPModel(StandardModel):
    """
    A wrapper for OpenCLIP Models
    """
    def __init__(self, model_name, device):
        self.device = device
        model = None
        preprocess = None

        if model_name in wc.OPEN_CLIP_MODELS:
            model, _, preprocess = open_clip.create_model_and_transforms(wc.OPEN_CLIP_MODELS[model_name])
        else:
            raise ValueError(f"Unknown model_name: {model_name}")


        model = model.to(device)
        super().__init__(model, model_name, transform=preprocess)
        with torch.no_grad():
            zeroshot_weights = []
            for class_name in tqdm(imagenet_classes):
                texts = [template(class_name) for template in openai_imagenet_template]  # format with class
                texts = open_clip.tokenize(texts).to(device)  # tokenize
                class_embeddings = self.model.encode_text(texts)  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            self._text_features = torch.stack(zeroshot_weights, dim=1).to(device)

    def __call__(self, image_input):
        image_features = self.model.encode_image(image_input)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ self._text_features).softmax(dim=-1)
        values = similarity
        return values


def download_file(url, base_dir=".", sub_dir="checkpoint", new_name=None):
    import wget
    """
    Downloads a file from a URL, ensures the directory structure exists,
    and handles renaming if the file already exists.
    
    Args:
        url (str): The URL of the file to download.
        base_dir (str): The base directory for saving files.
        sub_dir (str): Sub-directory where files should be saved.
        new_name (str): New name for the downloaded file (optional).
    """
    save_dir = os.path.join(base_dir, sub_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    temp_path = wget.download(url, out=save_dir)
    print(f"\nDownloaded to {temp_path}")

    if not new_name:
        new_name = os.path.basename(temp_path) 
    
    final_path = os.path.join(save_dir, new_name)

    if os.path.exists(final_path):
        return
    
    os.rename(temp_path, final_path)
    print(f"File saved as {final_path}")
