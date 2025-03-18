import timm.data
import torchvision.datasets as dset
import torch
import torchvision.transforms as tv_transforms
#from torchvision import transforms
import torchvision.models as models
import os
from robustbench import utils as rob_bench_utils
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import requests
from os.path import join
from tqdm import tqdm
import shutil
import bagnets.pytorchnet
import torch.nn as nn
from abc import abstractmethod
import quba_constants as con
import bcos.models.pretrained as bcosmodels
from bcos import transforms as bcos_transforms
import torch.nn.functional as F
from helper.imagenet import imagenet_templates as openai_imagenet_template
import torch
import quba_constants as wc

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

        model, preprocess = create_model_from_pretrained('hf-hub:timm/' + model_name)
        tokenizer = get_tokenizer('hf-hub:timm/' + model_name)

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

        if model_name in wc.MODEL_CONFIGS:
            model, _, preprocess = open_clip.create_model_and_transforms(wc.MODEL_CONFIGS[model_name])
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

def load_dataset(model, path, num_workers=8, batch_size=32):
    
    """
    Returns loader for the ImageNet Dataset
    
    :model: model of class StandardModel
    :path: path to dataset
    :batch_size: Batch Size of dataloader
    :num_workers: Number of Workers
    :return: ImageNet Dataloader
    """

    transform = None
    
    dataset = dset.ImageFolder(
                root=path, 
                transform=transform if transform is not None else model.transform
                )
    
    #dataset = MyImageFolder(
    #            root=path, 
    #            transform=transform if transform is not None else model.transform
    #            )

    dataset_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)

    return dataset_loader

def load_model(model_arch, args):
    """
    Loads model specified by model_arch

    :model_arch: model to load
    :return: returns loaded model
    """

    model = None
    model_name = model_arch
    transform = None
    timm_name = None
    head = None

    if "AlexNet" == model_arch:
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        transform = models.AlexNet_Weights.IMAGENET1K_V1.transforms()

    elif "GoogLeNet" == model_arch:
        model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        transform = models.GoogLeNet_Weights.IMAGENET1K_V1.transforms()

    elif "ResNet18" == model_arch:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    elif "ResNet34" == model_arch:
        model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        transform = models.ResNet34_Weights.IMAGENET1K_V1.transforms()
    elif "ResNet50" == model_arch:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        transform = models.ResNet50_Weights.IMAGENET1K_V1.transforms()
    elif "ResNet101" == model_arch:
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        transform = models.ResNet101_Weights.IMAGENET1K_V1.transforms()
    elif "ResNet152" == model_arch:
        model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        transform = models.ResNet152_Weights.IMAGENET1K_V1.transforms()

    #Bcos Models https://github.com/B-cos/B-cos-v2/tree/main    
    elif "bcos-ResNet18" == model_arch:
        model = bcosmodels.resnet18(pretrained=True)
        model_name = "bcos-ResNet18"
    elif "bcos-ResNet34" == model_arch:
        model = bcosmodels.resnet34(pretrained=True)
        model_name = "bcos-ResNet34"
    elif "bcos-ResNet50" == model_arch:
        model = bcosmodels.resnet50(pretrained=True)
        model_name = "bcos-ResNet50"
    elif "bcos-ResNet101" == model_arch:
        model = bcosmodels.resnet101(pretrained=True)
        model_name = "bcos-ResNet101"
    elif "bcos-ResNet152" == model_arch:
        model = bcosmodels.resnet152(pretrained=True)
        model_name = "bcos-ResNet152"
    elif "bcos-DenseNet121" == model_arch:
        model = bcosmodels.densenet121(pretrained=True)
        model_name = "bcos-DenseNet121"
    elif "bcos-DenseNet161" == model_arch:
        model = bcosmodels.densenet161(pretrained=True)
        model_name = "bcos-DenseNet161"
    elif "bcos-DenseNet169" == model_arch:
        model = bcosmodels.densenet169(pretrained=True)
        model_name = "bcos-DenseNet169"
    elif "bcos-DenseNet201" == model_arch:
        model = bcosmodels.densenet201(pretrained=True)
        model_name = "bcos-DenseNet201"
    elif "bcos-simple-vit-b-patch16-224" == model_arch:
        model = bcosmodels.simple_vit_b_patch16_224(pretrained=True)
        model_name = "bcos-simple-vit-b-patch16-224"
    elif "bcos-convnext-tiny" == model_arch:
        model = bcosmodels.convnext_tiny(pretrained=True)
        model_name = "bcos-convnext-tiny"
    elif "bcos-convnext-base" == model_arch:
        model = bcosmodels.convnext_base(pretrained=True)
        model_name = "bcos-convnext-base"

    elif "SqueezeNet" == model_arch:
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        transform = models.SqueezeNet1_1_Weights.IMAGENET1K_V1.transforms()

    elif "ResNeXt50-32x4d" == model_arch:
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        transform = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1.transforms()
    elif "ResNeXt101-32x8d" == model_arch:
        model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        transform = models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1.transforms()
    elif "ResNeXt101-64x4d" == model_arch:
        model = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
        transform = models.ResNeXt101_64X4D_Weights.IMAGENET1K_V1.transforms()

    elif "DenseNet121" == model_arch:
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        transform = models.DenseNet121_Weights.IMAGENET1K_V1.transforms()
    elif "DenseNet161" == model_arch:
        model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        transform = models.DenseNet161_Weights.IMAGENET1K_V1.transforms()
    elif "DenseNet169" == model_arch:
        model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
        transform = models.DenseNet169_Weights.IMAGENET1K_V1.transforms()
    elif "DenseNet201" == model_arch:
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        transform = models.DenseNet201_Weights.IMAGENET1K_V1.transforms()

    elif "BagNet9" == model_arch:
        model = bagnets.pytorchnet.bagnet9(pretrained=True)
    elif "BagNet17" == model_arch:
        model = bagnets.pytorchnet.bagnet17(pretrained=True)
    elif "BagNet33" == model_arch:
        model = bagnets.pytorchnet.bagnet33(pretrained=True)

    elif "VGG11" == model_arch:
        model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        transform = models.VGG11_Weights.IMAGENET1K_V1.transforms()
    elif "VGG13" == model_arch:
        model = models.vgg13(weights=models.VGG13_Weights.IMAGENET1K_V1)
        transform = models.VGG13_Weights.IMAGENET1K_V1.transforms()
    elif "VGG16" == model_arch:
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        transform = models.VGG16_Weights.IMAGENET1K_V1.transforms()
    elif "VGG19" == model_arch:
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        transform = models.VGG19_Weights.IMAGENET1K_V1.transforms()

    elif "VGG11-bn" == model_arch:
        model = models.vgg11_bn(weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
        transform = models.VGG11_BN_Weights.IMAGENET1K_V1.transforms()
    elif "VGG13-bn" == model_arch:
        model = models.vgg13_bn(weights=models.VGG13_BN_Weights.IMAGENET1K_V1)
        transform = models.VGG13_BN_Weights.IMAGENET1K_V1.transforms()
    elif "VGG16-bn" == model_arch:
        model = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1)
        transform = models.VGG16_BN_Weights.IMAGENET1K_V1.transforms()
    elif "VGG19-bn" == model_arch:
        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        transform = models.VGG19_BN_Weights.IMAGENET1K_V1.transforms()

    elif "ViT-b-16" == model_arch:
        model = models.vit_b_16( weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        transform = models.ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    elif "ViT-l-16" == model_arch:
        model = models.vit_l_16( weights=models.ViT_L_16_Weights.IMAGENET1K_V1)
        transform = models.ViT_L_16_Weights.IMAGENET1K_V1.transforms()
    elif "ViT-b-32" == model_arch:
        model = models.vit_b_32(weights=models.ViT_B_32_Weights.IMAGENET1K_V1)
        transform = models.ViT_B_32_Weights.IMAGENET1K_V1.transforms()
    elif "ViT-l-32" == model_arch:
        model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
        transform = models.ViT_L_32_Weights.IMAGENET1K_V1.transforms()

    elif "ViT-t5-16" == model_arch:
        timm_name = "tiny_vit_5m_224.in1k"
    elif "ViT-t11-16" == model_arch:
        timm_name = "tiny_vit_11m_224.in1k"
    elif "ViT-t21-16" == model_arch:
        timm_name = "tiny_vit_21m_224.in1k"
    elif "ViT-s-16" == model_arch:
        timm_name = "vit_small_patch16_224.augreg_in1k"

    elif "ViT-t5-16-21k" == model_arch:
        timm_name = "tiny_vit_5m_224.dist_in22k_ft_in1k"
    elif "ViT-t11-16-21k" == model_arch:
        timm_name = "tiny_vit_11m_224.dist_in22k_ft_in1k"
    elif "ViT-t21-16-21k" == model_arch:
        timm_name = "tiny_vit_21m_224.dist_in22k_ft_in1k"
    elif "ViT-s-16-21k" == model_arch:
        timm_name = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    elif "ViT-b-16-21k" == model_arch:
        timm_name = "vit_base_patch16_224.augreg_in21k_ft_in1k"
    elif "ViT-l-16-21k" == model_arch:
        timm_name = "vit_large_patch16_224.augreg_in21k_ft_in1k"
    elif "ViT-b-32-21k" == model_arch:
        timm_name = "vit_base_patch32_224.augreg_in21k_ft_in1k"
    
    elif "ViT-l-32-21k" == model_arch:
        timm_name = "vit_large_patch32_384.orig_in21k_ft_in1k"
    
    elif "Salman2020Do-RN50" == model_arch:
        model = rob_bench_utils.load_model(model_name='Salman2020Do_R50', dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
    elif "Salman2020Do-RN50-2" == model_arch:
        model = rob_bench_utils.load_model(model_name='Salman2020Do_50_2', dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")

    elif "Liu2023Comprehensive-Swin-L" == model_arch:
        model = rob_bench_utils.load_model(model_name="Liu2023Comprehensive_Swin-L", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
    elif "Liu2023Comprehensive-ConvNeXt-L" == model_arch:
        model = rob_bench_utils.load_model(model_name="Liu2023Comprehensive_ConvNeXt-L", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
    elif "Liu2023Comprehensive-Swin-B" == model_arch:
        model = rob_bench_utils.load_model(model_name="Liu2023Comprehensive_Swin-B", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
    elif "Liu2023Comprehensive-ConvNeXt-B" == model_arch:
        model = rob_bench_utils.load_model(model_name="Liu2023Comprehensive_ConvNeXt-B", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")

    elif "Singh2023Revisiting-ConvNeXt-L-ConvStem" == model_arch:
        model = rob_bench_utils.load_model(model_name="Singh2023Revisiting_ConvNeXt-L-ConvStem", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
    elif "Singh2023Revisiting-ConvNeXt-B-ConvStem" == model_arch:
        model = rob_bench_utils.load_model(model_name="Singh2023Revisiting_ConvNeXt-B-ConvStem", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
    elif "Singh2023Revisiting-ViT-B-ConvStem" == model_arch:
        model = rob_bench_utils.load_model(model_name="Singh2023Revisiting_ViT-B-ConvStem", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
    elif "Singh2023Revisiting-ViT-S-ConvStem" == model_arch:
        model = rob_bench_utils.load_model(model_name="Singh2023Revisiting_ViT-S-ConvStem", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
    elif "Singh2023Revisiting-ConvNeXt-S-ConvStem" == model_arch:
        model = rob_bench_utils.load_model(model_name="Singh2023Revisiting_ConvNeXt-S-ConvStem", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
    elif "Singh2023Revisiting-ConvNeXt-T-ConvStem" == model_arch:
        model = rob_bench_utils.load_model(model_name="Singh2023Revisiting_ConvNeXt-T-ConvStem", dataset=rob_bench_utils.BenchmarkDataset.imagenet, threat_model="Linf")
        
    #Big-Transfer (BiT) Models: https://arxiv.org/abs/1912.11370
    elif "BiTM-resnetv2-50x1" == model_arch: 
        timm_name = "resnetv2_50x1_bit.goog_in21k_ft_in1k"
    elif "BiTM-resnetv2-50x3"== model_arch:
        timm_name = "resnetv2_50x3_bitm"
    elif "BiTM-resnetv2-101x1" == model_arch:
        timm_name = "resnetv2_101x1_bit.goog_in21k_ft_in1k"
    elif "BiTM-resnetv2-152x2"== model_arch:
        timm_name = "resnetv2_152x2_bitm"

    elif "WRN-50-2" == model_arch:
        model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        transform = models.Wide_ResNet50_2_Weights.IMAGENET1K_V1.transforms()
    elif "WRN-101-2" == model_arch:
        model = models.wide_resnet101_2(weights=models.Wide_ResNet101_2_Weights.IMAGENET1K_V1)
        transform = models.Wide_ResNet101_2_Weights.IMAGENET1K_V1.transforms()

    elif "Swin-T" == model_arch:
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        transform = models.Swin_T_Weights.IMAGENET1K_V1.transforms()
    elif "Swin-S" == model_arch:
        model = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1)
        transform = models.Swin_S_Weights.IMAGENET1K_V1.transforms()
    elif "Swin-B" == model_arch:
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
        transform = models.Swin_B_Weights.IMAGENET1K_V1.transforms()

    elif "SwinV2-T-W8" == model_arch:
        model = models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1)
        transform = models.Swin_V2_T_Weights.IMAGENET1K_V1.transforms()
    elif "SwinV2-S-W8" == model_arch:
        model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
        transform = models.Swin_V2_S_Weights.IMAGENET1K_V1.transforms()
    elif "SwinV2-B-W8" == model_arch:
        model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.IMAGENET1K_V1)
        transform = models.Swin_V2_B_Weights.IMAGENET1K_V1.transforms()                    

    elif "ConvNext-T" == model_arch:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        transform = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    elif "ConvNext-S" == model_arch:
        model = models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1)
        transform = models.ConvNeXt_Small_Weights.IMAGENET1K_V1.transforms()
    elif "ConvNext-B" == model_arch:
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
        transform = models.ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms()
    elif "ConvNext-L" == model_arch:
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
        transform = models.ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms()
    
    elif "ConvNext-T-21k" == model_arch:
        timm_name = "convnext_tiny.fb_in22k_ft_in1k"
    elif "ConvNext-S-21k" == model_arch:
        timm_name = "convnext_small.fb_in22k_ft_in1k"
    elif "ConvNext-B-21k" == model_arch:
        timm_name = "convnext_base.fb_in22k_ft_in1k"
    elif "ConvNext-L-21k" == model_arch:
        timm_name = "convnext_large.fb_in22k_ft_in1k"
    
    elif "ConvNextV2-N" == model_arch:
        timm_name = "convnextv2_nano.fcmae_ft_in1k"
    elif "ConvNextV2-T" == model_arch:
        timm_name = "convnextv2_tiny.fcmae_ft_in1k"
    elif "ConvNextV2-B" == model_arch:
        timm_name = "convnextv2_base.fcmae_ft_in1k"
    elif "ConvNextV2-L" == model_arch:
        timm_name = "convnextv2_large.fcmae_ft_in1k"

    elif "ConvNextV2-N-21k" == model_arch:
        timm_name = "convnextv2_nano.fcmae_ft_in22k_in1k"
    elif "ConvNextV2-T-21k" == model_arch:
        timm_name = "convnextv2_tiny.fcmae_ft_in22k_in1k"
    elif "ConvNextV2-B-21k" == model_arch:
        timm_name = "convnextv2_base.fcmae_ft_in22k_in1k"
    elif "ConvNextV2-L-21k" == model_arch:
        timm_name = "convnextv2_large.fcmae_ft_in22k_in1k"

    elif "Hiera-T" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_tiny_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
    elif "Hiera-S" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_small_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
    elif "Hiera-B" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
    elif "Hiera-B-Plus" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_base_plus_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")     
    elif "Hiera-L" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_large_224", pretrained=True, checkpoint="mae_in1k_ft_in1k")
    
    elif "EfficientNet-v2-S" == model_arch:
        timm_name = "tf_efficientnetv2_s.in1k"
    elif "EfficientNet-v2-M" == model_arch:
        timm_name = "tf_efficientnetv2_m.in1k"
    elif "EfficientNet-v2-L" == model_arch:
        timm_name = "tf_efficientnetv2_l.in1k"

    elif "EfficientNet-v2-S-21k" == model_arch:
        timm_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    elif "EfficientNet-v2-M-21k" == model_arch:
        timm_name = "tf_efficientnetv2_m.in21k_ft_in1k"
    elif "EfficientNet-v2-L-21k" == model_arch:
        timm_name = "tf_efficientnetv2_l.in21k_ft_in1k"

    elif "InceptionV3" == model_arch:
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        #model.aux_logits = False
        transform = models.Inception_V3_Weights.IMAGENET1K_V1.transforms()
    elif "InceptionV4" == model_arch:
        timm_name = "inception_v4.tf_in1k"

    elif "MobileNetV2" == model_arch:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        transform = models.MobileNet_V2_Weights.IMAGENET1K_V1.transforms()
    elif "MobileNetV3-s" == model_arch:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        transform = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    elif "MobileNetV3-l" == model_arch:
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        transform = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1.transforms()

    elif "NS-EfficientNet-B0" == model_arch:
        timm_name = "tf_efficientnet_b0.ns_jft_in1k"
    elif "NS-EfficientNet-B1" == model_arch:
        timm_name = "tf_efficientnet_b1.ns_jft_in1k"
    elif "NS-EfficientNet-B2" == model_arch:
        timm_name = "tf_efficientnet_b2.ns_jft_in1k"
    elif "NS-EfficientNet-B3" == model_arch:
        timm_name = "tf_efficientnet_b3.ns_jft_in1k"
    elif "NS-EfficientNet-B4" == model_arch:
        timm_name = "tf_efficientnet_b4.ns_jft_in1k"
    elif "NS-EfficientNet-B5" == model_arch:
        timm_name = "tf_efficientnet_b5.ns_jft_in1k"
    elif "NS-EfficientNet-B6" == model_arch:
        timm_name = "tf_efficientnet_b6.ns_jft_in1k"
    elif "NS-EfficientNet-B7" == model_arch:
        timm_name = "tf_efficientnet_b7.ns_jft_in1k"
    
    elif "DeiT-t" == model_arch:
        timm_name = "deit_tiny_patch16_224.fb_in1k"
    elif "DeiT-s" == model_arch:
        timm_name = "deit_small_patch16_224.fb_in1k"
    elif "DeiT-b" == model_arch:
        timm_name = "deit_base_patch16_224.fb_in1k"
    elif "DeiT3-s" == model_arch:
        timm_name = "deit3_small_patch16_224.fb_in1k"
    elif "DeiT3-m" == model_arch:
        timm_name = "deit3_medium_patch16_224.fb_in1k"
    elif "DeiT3-b" == model_arch:
        timm_name = "deit3_base_patch16_224.fb_in1k"
    elif "DeiT3-l" == model_arch:
        timm_name = "deit3_large_patch16_224.fb_in1k"

    elif "DeiT3-s-21k" == model_arch:
        timm_name = "deit3_small_patch16_224.fb_in22k_ft_in1k"
    elif "DeiT3-m-21k" == model_arch:
        timm_name = "deit3_medium_patch16_224.fb_in22k_ft_in1k"
    elif "DeiT3-b-21k" == model_arch:
        timm_name = "deit3_base_patch16_224.fb_in22k_ft_in1k"
    elif "DeiT3-l-21k" == model_arch:
        timm_name = "deit3_large_patch16_224.fb_in22k_ft_in1k"

    elif "MaxViT-t" == model_arch:
        model = models.maxvit_t(weights=models.MaxVit_T_Weights.IMAGENET1K_V1)
        transform = models.MaxVit_T_Weights.IMAGENET1K_V1.transforms()
    elif "MaxViT-b" == model_arch:
        timm_name = "maxvit_base_tf_224.in1k"
    elif "MaxViT-l" == model_arch:
        timm_name = "maxvit_large_tf_224.in1k"

    elif "CrossViT-9dagger" == model_arch:
        timm_name = "crossvit_9_dagger_240.in1k"
    elif "CrossViT-15dagger" == model_arch:
        timm_name = "crossvit_15_dagger_240.in1k"
    elif "CrossViT-18dagger" == model_arch:
        timm_name = "crossvit_18_dagger_240.in1k"

    elif "FastViT-sa12" == model_arch:
        timm_name = "fastvit_sa12.apple_in1k"
    elif "FastViT-sa24" == model_arch:
        timm_name = "fastvit_sa24.apple_in1k"
    elif "FastViT-sa36" == model_arch:
        timm_name = "fastvit_sa36.apple_in1k"

    elif "XCiT-s24-16" == model_arch:
        timm_name = "xcit_small_24_p16_224.fb_in1k"
    elif "XCiT-m24-16" == model_arch:
        timm_name = "xcit_medium_24_p16_224.fb_in1k"
    elif "XCiT-l24-16" == model_arch:
        timm_name = "xcit_large_24_p16_224.fb_in1k"


    elif "LeViT-128" == model_arch:
        timm_name = "levit_128.fb_dist_in1k"
    elif "LeViT-256" == model_arch:
        timm_name = "levit_256.fb_dist_in1k"
    elif "LeViT-384" == model_arch:
        timm_name = "levit_384.fb_dist_in1k"
    
    elif "MViTv2-t" == model_arch:
        timm_name = "mvitv2_tiny.fb_in1k"
    elif "MViTv2-s" == model_arch:
        timm_name = "mvitv2_small.fb_in1k"
    elif "MViTv2-b" == model_arch:
        timm_name = "mvitv2_base.fb_in1k"
    elif "MViTv2-l" == model_arch:
        timm_name = "mvitv2_large.fb_in1k"

    elif "BeiT-b" == model_arch:
        timm_name = "beit_base_patch16_224.in22k_ft_in22k_in1k"
    
    elif "ConViT-t" == model_arch:
        timm_name = "convit_tiny.fb_in1k"
    elif "ConViT-s" == model_arch:
        timm_name = "convit_small.fb_in1k"
    elif "ConViT-b" == model_arch:
        timm_name = "convit_base.fb_in1k"

    elif "CaiT-xxs24" == model_arch:
        timm_name = "cait_xxs24_384.fb_dist_in1k"
    elif "CaiT-xs24" == model_arch:
        timm_name = "cait_xs24_384.fb_dist_in1k"
    elif "CaiT-s24" == model_arch:
        timm_name = "cait_s24_384.fb_dist_in1k"

    elif "EVA02-t-21k" == model_arch:
        timm_name = "eva02_tiny_patch14_336.mim_in22k_ft_in1k"
    elif "EVA02-s-21k" == model_arch:
        timm_name = "eva02_small_patch14_336.mim_in22k_ft_in1k"
    elif "EVA02-b-21k" == model_arch:
        timm_name = "eva02_base_patch14_448.mim_in22k_ft_in1k"
    
    elif "Inception-ResNetv2" == model_arch:
        timm_name = "inception_resnet_v2.tf_in1k"

    elif "SwinV2-t-W16" == model_arch:
        timm_name = "swinv2_tiny_window16_256.ms_in1k"
    elif "SwinV2-s-Win16" == model_arch:  
        timm_name = "swinv2_small_window16_256.ms_in1k"
    elif "SwinV2-b-Win16" == model_arch:  
        timm_name = "swinv2_base_window16_256.ms_in1k"
    elif "SwinV2-b-Win12to16-21k" == model_arch:
        timm_name = "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"
    elif "SwinV2-l-Win12to16-21k" == model_arch:
        timm_name = "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k"

    elif "MnasNet-05" == model_arch:
        model = models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1)
        transform = models.MNASNet0_5_Weights.IMAGENET1K_V1.transforms()
    elif "MnasNet-075" == model_arch:
        model = models.mnasnet0_75(weights=models.MNASNet0_75_Weights.IMAGENET1K_V1)
        transform = models.MNASNet0_75_Weights.IMAGENET1K_V1.transforms()
    elif "MnasNet-1" == model_arch:
        model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.IMAGENET1K_V1)
        transform = models.MNASNet1_0_Weights.IMAGENET1K_V1.transforms()
    elif "MnasNet-13" == model_arch:
        model = models.mnasnet1_3(weights=models.MNASNet1_3_Weights.IMAGENET1K_V1)
        transform = models.MNASNet1_3_Weights.IMAGENET1K_V1.transforms()
    
    elif "ShuffleNet-v2-05" == model_arch:
        model = models.shufflenet_v2_x0_5(weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
        transform = models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1.transforms()
    elif "ShuffleNet-v2-1" == model_arch:
        model = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        transform = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1.transforms()
    elif "ShuffleNet-v2-15" == model_arch:
        model = models.shufflenet_v2_x1_5(weights=models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1)
        transform = models.ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1.transforms()
    elif "ShuffleNet-v2-2" == model_arch:
        model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
        transform = models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1.transforms()
        
    elif "InceptionNext-t" == model_arch:
        timm_name = "inception_next_tiny.sail_in1k"
    elif "InceptionNext-s" == model_arch:
        timm_name = "inception_next_small.sail_in1k"
    elif "InceptionNext-b" == model_arch:
        timm_name = "inception_next_base.sail_in1k" 

    elif "Xception" == model_arch:
        timm_name = "legacy_xception.tf_in1k"

    elif "NasNet-l" == model_arch:
        timm_name = "nasnetalarge.tf_in1k"

    elif "PiT-t" == model_arch:
        timm_name = "pit_ti_224.in1k"
    elif "PiT-xs" == model_arch:
        timm_name = "pit_xs_224.in1k"
    elif "PiT-s" == model_arch:
        timm_name = "pit_s_224.in1k"
    elif "PiT-b" == model_arch:
        timm_name = "pit_b_224.in1k"

    elif "EfficientFormer-l1" == model_arch:
        timm_name = "efficientformer_l1.snap_dist_in1k"
    elif "EfficientFormer-l3" == model_arch:
        timm_name = "efficientformer_l3.snap_dist_in1k"
    elif "EfficientFormer-l7" == model_arch:
        timm_name = "efficientformer_l7.snap_dist_in1k"

    elif "MobileNetV3-l-21k" == model_arch:
        timm_name = "mobilenetv3_large_100.miil_in21k_ft_in1k"

    elif "DaViT-t" == model_arch:
        timm_name = "davit_tiny.msft_in1k"
    elif "DaViT-s" == model_arch:
        timm_name = "davit_small.msft_in1k"
    elif "DaViT-b" == model_arch:
        timm_name = "davit_base.msft_in1k"
    
    elif "CoaT-t-lite" == model_arch:
        timm_name = "coat_lite_tiny.in1k"
    elif "CoaT-mi-lite" == model_arch:
        timm_name = "coat_lite_mini.in1k"
    elif "CoaT-s-lite" == model_arch:
        timm_name = "coat_lite_small.in1k"
    elif "CoaT-me-lite" == model_arch:
        timm_name = "coat_lite_medium.in1k"

    elif "RegNet-y-400mf" == model_arch:
        model = models.regnet_y_400mf(weights=models.RegNet_Y_400MF_Weights.IMAGENET1K_V1)
        transform = models.RegNet_Y_400MF_Weights.IMAGENET1K_V1.transforms()
    elif "RegNet-y-800mf" == model_arch:
        model = models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V1)
        transform = models.RegNet_Y_800MF_Weights.IMAGENET1K_V1.transforms()
    elif "RegNet-y-1-6gf" == model_arch:
        model = models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1)
        transform = models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1.transforms()
    elif "RegNet-y-3-2gf" == model_arch:
        model = models.regnet_y_3_2gf(weights=models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1)
        transform = models.RegNet_Y_3_2GF_Weights.IMAGENET1K_V1.transforms()
    elif "RegNet-y-8gf" == model_arch:
        model = models.regnet_y_8gf(weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V1)
        transform = models.RegNet_X_8GF_Weights.IMAGENET1K_V1.transforms()
    elif "RegNet-y-16gf" == model_arch:
        model = models.regnet_y_16gf(weights=models.RegNet_Y_16GF_Weights.IMAGENET1K_V1)
        transform = models.RegNet_Y_16GF_Weights.IMAGENET1K_V1.transforms()
    elif "RegNet-y-32gf" == model_arch:
        model = models.regnet_y_32gf(weights=models.RegNet_Y_32GF_Weights.IMAGENET1K_V1)
        transform = models.RegNet_Y_32GF_Weights.IMAGENET1K_V1.transforms()


    elif model_arch == "ResNet50-a1":
        timm_name = "resnet50.a1_in1k"
    elif model_arch == "ResNet50-ig1B":
        timm_name = "resnet50.fb_swsl_ig1b_ft_in1k"
    elif model_arch == "ResNet50-yfcc100m":
        timm_name = "resnet50.fb_ssl_yfcc100m_ft_in1k"


    #A1
    elif model_arch == "ResNet18-A1":
        timm_name = "resnet18.a1_in1k"
    elif model_arch == "ResNext50_32x4d_A1":
        timm_name = "resnext50_32x4d.a1_in1k"
    elif model_arch == "ResNet34-A1":
        timm_name = "resnet34.a1_in1k"
    elif model_arch == "ResNet101-A1":
        timm_name = "resnet101.a1_in1k"
    elif model_arch == "ResNet152-A1":
        timm_name = "resnet152.a1_in1k"

    elif model_arch == "BeiTV2-b":
        timm_name = "beitv2_base_patch16_224.in1k_ft_in1k"
    elif model_arch == "ViT-b-14-dinoV2-LP":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
    
    elif model_arch == "ViT-s-14-dinoV2-LP":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)

    elif model_arch == "ViT-l-14-dinoV2-LP":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)

    elif "ResNet50-DINO-LP" == model_arch:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_resnet50_pretrain/dino_resnet50_linearweights.pth")["state_dict"]
        new_ckpt = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.linear.', '')  
            new_ckpt[new_key] = value
        model.fc = nn.Linear(in_features=2048, out_features=1000)
        model.fc.load_state_dict(new_ckpt, strict=True)
        transform = tv_transforms.Compose([
                tv_transforms.Resize(256, interpolation=3),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    elif "ViT-b-16-DINO-LP" == model_arch:
        import helper.models.dino.vision_transformer as dino_vit
        model = dino_vit.__dict__["vit_base"](patch_size=16, num_classes=0)
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth")
        model.load_state_dict(state_dict)
        ckpt = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + "dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth")["state_dict"]
        new_ckpt = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.linear.', '')  
            new_ckpt[new_key] = value
                
        head = nn.Linear(in_features=1536, out_features=1000)
        head.load_state_dict(new_ckpt, strict=True)
        transform = tv_transforms.Compose([
                tv_transforms.Resize(256, interpolation=3),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    elif "ViT-s-16-DINO-LP" == model_arch:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16') #dino.vision_transformer.__dict__["vit_small"](patch_size=16, num_classes=0)
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth")
        model.load_state_dict(state_dict)
        ckpt = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth")["state_dict"]
        new_ckpt = {}
        for key, value in ckpt.items():
            new_key = key.replace('module.linear.', '')  
            new_ckpt[new_key] = value
                
        head = nn.Linear(in_features=1536, out_features=1000)
        head.load_state_dict(new_ckpt, strict=True)
        transform = tv_transforms.Compose([
                tv_transforms.Resize(256, interpolation=3),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    elif "vit-t-16-21k" == model_arch:
        timm_name = "vit_tiny_patch16_224.augreg_in21k_ft_in1k"
    
    elif model_arch == "vit-b-16-mae-lp":
        import helper.models.mae.models_vit as models_vit
        model = models_vit.vit_base_patch16()
        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
        ckpt = torch.load(wc.MAE_LP)
        model.load_state_dict(ckpt["model"])
        transform = tv_transforms.Compose([
            tv_transforms.Resize(256, interpolation=3),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    elif model_arch == "ResNet50-DINO-FT":
        import helper.models.dino.resnet_dino
        model = helper.models.dino.resnet_dino.resnet50_dino()
        ckpt = torch.load(wc.DINO_RESNET_FT)
        model.load_state_dict(ckpt["state_dict"])
        transform = tv_transforms.Compose([
                tv_transforms.Resize(256, interpolation=3),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif model_arch == "ViTB-DINO-FT":
        model = timm.create_model("timm/vit_base_patch16_224.dino")
        model.head = nn.Linear(in_features=768, out_features=1000, bias=True)
        ckpt = torch.load(wc.DINO_VIT_FT)
        model.load_state_dict(ckpt["state_dict"])
        transform = tv_transforms.Compose([
                tv_transforms.Resize(256, interpolation=3),
                tv_transforms.CenterCrop(224),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    
    elif model_arch == "vit-b-16-mae-ft":
        import helper.models.mae.models_vit as models_vit
        model = models_vit.vit_base_patch16(num_classes=1000, drop_path_rate=0.1, global_pool=True)
        download_file(url="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth", new_name="mae_finetuned_vit_base.pth")
        ckpt = torch.load(wc.MAE_FT)["model"]
        model.load_state_dict(ckpt)
        transform = tv_transforms.Compose([
            tv_transforms.Resize(256, interpolation=3),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    elif model_arch == "ResNeXt101-32x8d-IG1B":
        timm_name = "resnext101_32x8d.fb_swsl_ig1b_ft_in1k"
    elif model_arch == "ResNeXt50-32x4d-YFCCM100":
        timm_name = "resnext50_32x4d.fb_ssl_yfcc100m_ft_in1k"
    elif model_arch == "ResNeXt50-32x4d-IG1B":
        timm_name = "resnext50_32x4d.fb_swsl_ig1b_ft_in1k"
    elif model_arch == "ResNet18-IG1B":
        timm_name = "resnet18.fb_swsl_ig1b_ft_in1k"


    elif model_arch == 'EfficientNet-b0-A1':
        timm_name = "tf_efficientnet_b0.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b0_a1_0-9188dd46.pth",new_name="tf_efficientnet_b0_a1.pth")
        model = timm.create_model("tf_efficientnet_b0.in1k", wc.RSB_ROOT + "tf_efficientnet_b0_a1.pth")
    elif model_arch == 'EfficientNet-b1-A1':
        timm_name = "tf_efficientnet_b1.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b1_a1_0-b55e845c.pth", new_name="tf_efficientnet_b1_a1.pth")
        model = timm.create_model("tf_efficientnet_b1.in1k", wc.RSB_ROOT + "tf_efficientnet_b1_a1.pth")
    elif model_arch == 'EfficientNet-b2-A1':
        timm_name = "tf_efficientnet_b2.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b2_a1_0-f1382665.pth", new_name="tf_efficientnet_b2_a1.pth")
        model = timm.create_model("tf_efficientnet_b2.in1k", wc.RSB_ROOT + "tf_efficientnet_b2_a1.pth")
    elif model_arch == 'EfficientNet-b3-A1':
        timm_name = "tf_efficientnet_b3.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b3_a1_0-efc81b92.pth", new_name="tf_efficientnet_b3_a1.pth")
        model = timm.create_model("tf_efficientnet_b3.in1k", wc.RSB_ROOT + "tf_efficientnet_b3_a1.pth")
    elif model_arch == 'EfficientNet-b4-A1':
        timm_name = "tf_efficientnet_b4.in1k"
        download_file(url=wc.RSB_LINK + "f_efficientnet_b4_a1_0-182bef54.pth", new_name="tf_efficientnet_b4_a1.pth")
        model = timm.create_model("tf_efficientnet_b4.in1k", wc.RSB_ROOT + "tf_efficientnet_b4_a1.pth")
    
    elif model_arch == 'EfficientNetv2-M-A1':
        timm_name = "efficientnetv2_rw_m.agc_in1k"
        download_file(url=wc.RSB_LINK + "efficientnetv2_rw_m_a1_0-b788290c.pth", new_name="efficientnetv2_rw_m_a1.pth")
        model = timm.create_model("efficientnetv2_rw_m.agc_in1k", wc.RSB_ROOT + "efficientnetv2_rw_m_a1.pth")
    elif model_arch == 'EfficientNetv2-S-A1':
        timm_name = "efficientnetv2_rw_s.ra2_in1k"
        download_file(url=wc.RSB_LINK + "efficientnetv2_rw_s_a1_0-59d76611.pth", new_name="efficientnetv2_rw_s_a1.pth")
        model = timm.create_model("efficientnetv2_rw_s.ra2_in1k", wc.RSB_ROOT + "efficientnetv2_rw_s_a1.pth")
    

    elif model_arch == 'RegNety-040-A1':
        timm_name = "regnety_040.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_040_a1_0-453380cb.pth", new_name="regnety_040_a1.pth")
        model = timm.create_model("regnety_040.pycls_in1k", wc.RSB_ROOT + "regnety_040_a1.pth")
    elif model_arch == 'RegNety-080-A1':
        timm_name = "regnety_080.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_080_a1_0-7d647454.pth", new_name="regnety_080_a1.pth")
        model = timm.create_model("regnety_080.pycls_in1k", wc.RSB_ROOT + "regnety_080_a1.pth")
    elif model_arch == 'RegNety-160-A1':
        timm_name = "regnety_160.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_160_a1_0-ed74711e.pth", new_name="regnety_160_a1.pth")
        model = timm.create_model("regnety_160.pycls_in1k", wc.RSB_ROOT + "regnety_160_a1.pth")
    elif model_arch == 'RegNety-320-A1':
        timm_name = "regnety_320.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_320_a1_0-6c920aed.pth", new_name="regnety_320_a1.pth")
        model = timm.create_model("regnety_320.pycls_in1k", wc.RSB_ROOT + "regnety_320_a1.pth")

    elif model_arch == 'ResNet18-A1':
        timm_name = "resnet18"
        download_file(url=wc.RSB_LINK + "resnet18_a1_0-d63eafa0.pth", new_name="resnet18_a1.pth")
        model = timm.create_model("resnet18", wc.RSB_ROOT + "resnet18_a1.pth")
    elif model_arch == 'ResNet34-A1':
        timm_name = "resnet34"
        download_file(url=wc.RSB_LINK + "resnet34_a1_0-46f8f793.pth", new_name="resnet34_a1.pth")
        model = timm.create_model("resnet34", wc.RSB_ROOT + "resnet34_a1.pth")
    elif model_arch == 'ResNet50-A1':
        timm_name = "resnet50"
        download_file(url=wc.RSB_LINK + "resnet50_a1_0-14fe96d1.pth", new_name="resnet50_a1.pth")
        model = timm.create_model("resnet50", wc.RSB_ROOT + "resnet50_a1.pth")
    elif model_arch == 'ResNet50d-A1':
        timm_name = "resnet50d.gluon_in1k"
        download_file(url=wc.RSB_LINK + "resnet50d_a1_0-e20cff14.pth", new_name="resnet50d_a1.pth")
        model = timm.create_model("resnet50d.gluon_in1k", wc.RSB_ROOT + "resnet50d_a1.pth")
    elif model_arch == 'ResNet101-A1':
        timm_name = "resnet101"
        download_file(url=wc.RSB_LINK + "resnet101_a1_0-cdcb52a9.pth", new_name="resnet101_a1.pth")
        model = timm.create_model("resnet101", wc.RSB_ROOT + "resnet101_a1.pth")
    elif model_arch == 'ResNet152-A1':
        timm_name = "resnet152"
        download_file(url=wc.RSB_LINK + "resnet152_a1_0-2eee8a7a.pth", new_name="resnet152_a1.pth")
        model = timm.create_model("resnet152", wc.RSB_ROOT + "resnet152_a1.pth")

    elif model_arch == 'ResNext50-32x4d-A1':
        timm_name = "resnext50_32x4d.gluon_in1k"
        download_file(url=wc.RSB_LINK + "resnext50_32x4d_a1_0-b5a91a1d.pth", new_name="resnext50_32x4d_a1.pth")
        model = timm.create_model("resnext50_32x4d.gluon_in1k", wc.RSB_ROOT + "resnext50_32x4d_a1.pth")
    elif model_arch == 'SeNet154-A1':
        timm_name = "senet154.gluon_in1k"
        download_file(url=wc.RSB_LINK + "gluon_senet154_a1_0-ef9d383e.pth", new_name="gluon_senet154_a1.pth")
        model = timm.create_model("senet154.gluon_in1k", wc.RSB_ROOT + "gluon_senet154_a1.pth")
    
    elif model_arch == 'EfficientNet-b0-A3':
        timm_name = "tf_efficientnet_b0.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b0_a3_0-94e799dc.pth",new_name="tf_efficientnet_b0_a3.pth")
        model = timm.create_model("tf_efficientnet_b0.in1k", wc.RSB_ROOT + "tf_efficientnet_b0_a3.pth")
    elif model_arch == 'EfficientNet-b1-A3':
        timm_name = "tf_efficientnet_b1.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b1_a3_0-ee9f9669.pth", new_name="tf_efficientnet_b1_a3.pth")
        model = timm.create_model("tf_efficientnet_b1.in1k", wc.RSB_ROOT + "tf_efficientnet_b1_a3.pth")
    elif model_arch == 'EfficientNet-b2-A3':
        timm_name = "tf_efficientnet_b2.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b2_a3_0-61f0f688.pth", new_name="tf_efficientnet_b2_a3.pth")
        model = timm.create_model("tf_efficientnet_b2.in1k", wc.RSB_ROOT + "tf_efficientnet_b2_a3.pth")
    elif model_arch == 'EfficientNet-b3-A3':
        timm_name = "tf_efficientnet_b3.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b3_a3_0-0a50fa9a.pth", new_name="tf_efficientnet_b3_a3.pth")
        model = timm.create_model("tf_efficientnet_b3.in1k", wc.RSB_ROOT + "tf_efficientnet_b3_a3.pth")
    elif model_arch == 'EfficientNet-b4-A3':
        timm_name = "tf_efficientnet_b4.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b4_a3_0-a6a8179a.pth", new_name="tf_efficientnet_b4_a3.pth")
        model = timm.create_model("tf_efficientnet_b4.in1k", wc.RSB_ROOT + "tf_efficientnet_b4_a3.pth")
    
    elif model_arch == 'EfficientNetv2-M-A3':
        timm_name = "efficientnetv2_rw_m.agc_in1k"
        download_file(url=wc.RSB_LINK + "efficientnetv2_rw_m_a3_0-68b15d26.pth", new_name="efficientnetv2_rw_m_a3.pth")
        model = timm.create_model("efficientnetv2_rw_m.agc_in1k", wc.RSB_ROOT + "efficientnetv2_rw_m_a3.pth")
    elif model_arch == 'EfficientNetv2-S-A3':
        timm_name = "efficientnetv2_rw_s.ra2_in1k"
        download_file(url=wc.RSB_LINK + "efficientnetv2_rw_s_a3_0-11105c48.pth", new_name="efficientnetv2_rw_s_a3.pth")
        model = timm.create_model("efficientnetv2_rw_s.ra2_in1k", wc.RSB_ROOT + "efficientnetv2_rw_s_a3.pth")
    

    elif model_arch == 'RegNety-040-A3':
        timm_name = "regnety_040.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_040_a3_0-9705a0d6.pth", new_name="regnety_040_a3.pth")
        model = timm.create_model("regnety_040.pycls_in1k", wc.RSB_ROOT + "regnety_040_a3.pth")
    elif model_arch == 'RegNety-080-A3':
        timm_name = "regnety_080.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_080_a3_0-2fb073a0.pth", new_name="regnety_080_a3.pth")
        model = timm.create_model("regnety_080.pycls_in1k", wc.RSB_ROOT + "regnety_080_a3.pth")
    elif model_arch == 'RegNety-160-A3':
        timm_name = "regnety_160.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_160_a3_0-9ee45d21.pth", new_name="regnety_160_a3.pth")
        model = timm.create_model("regnety_160.pycls_in1k", wc.RSB_ROOT + "regnety_160_a3.pth")
    elif model_arch == 'RegNety-320-A3':
        timm_name = "regnety_320.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_320_a3_0-242d2987.pth", new_name="regnety_320_a3.pth")
        model = timm.create_model("regnety_320.pycls_in1k", wc.RSB_ROOT + "regnety_320_a3.pth")

    elif model_arch == 'ResNet18-A3':
        timm_name = "resnet18"
        download_file(url=wc.RSB_LINK + "resnet18_a3_0-40c531c8.pth", new_name="resnet18_a3.pth")
        model = timm.create_model("resnet18", wc.RSB_ROOT + "resnet18_a3.pth")
    elif model_arch == 'ResNet34-A3':
        timm_name = "resnet34"
        download_file(url=wc.RSB_LINK + "resnet34_a3_0-a20cabb6.pth", new_name="resnet34_a3.pth")
        model = timm.create_model("resnet34", wc.RSB_ROOT + "resnet34_a3.pth")
    elif model_arch == 'ResNet50-A3':
        timm_name = "resnet50"
        download_file(url=wc.RSB_LINK + "resnet50_a3_0-59cae1ef.pth", new_name="resnet50_a3.pth")
        model = timm.create_model("resnet50", wc.RSB_ROOT + "resnet50_a3.pth")
    elif model_arch == 'ResNet50d-A3':
        timm_name = "resnet50d.gluon_in1k"
        download_file(url=wc.RSB_LINK + "resnet50d_a3_0-403fdfad.pth", new_name="resnet50d_a3.pth")
        model = timm.create_model("resnet50d.gluon_in1k", wc.RSB_ROOT + "resnet50d_a3.pth")
    elif model_arch == 'ResNet101-A3':
        timm_name = "resnet101"
        download_file(url=wc.RSB_LINK + "resnet101_a3_0-1db14157.pth", new_name="resnet101_a3.pth")
        model = timm.create_model("resnet101", wc.RSB_ROOT + "resnet101_a3.pth")
    elif model_arch == 'ResNet152-A3':
        timm_name = "resnet152"
        download_file(url=wc.RSB_LINK + "resnet152_a3_0-134d4688.pth", new_name="resnet152_a3.pth")
        model = timm.create_model("resnet152", wc.RSB_ROOT + "resnet152_a3.pth")

    elif model_arch == 'ResNext50-32x4d-A3':
        timm_name = "resnext50_32x4d.gluon_in1k"
        download_file(url=wc.RSB_LINK + "resnext50_32x4d_a3_0-3e450271.pth", new_name="resnext50_32x4d_a3.pth")
        model = timm.create_model("resnext50_32x4d.gluon_in1k", wc.RSB_ROOT + "resnext50_32x4d_a3.pth")
    elif model_arch == 'SeNet154-A3':
        timm_name = "senet154.gluon_in1k"
        download_file(url=wc.RSB_LINK + "gluon_senet154_a3_0-d8df0fde.pth", new_name="gluon_senet154_a3.pth")
        model = timm.create_model("senet154.gluon_in1k", wc.RSB_ROOT + "gluon_senet154_a3.pth")
    
    elif model_arch == 'EfficientNet-b0-A2':
        timm_name = "tf_efficientnet_b0.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b0_a2_0-48bede62.pth",new_name="tf_efficientnet_b0_a2.pth")
        model = timm.create_model("tf_efficientnet_b0.in1k", wc.RSB_ROOT + "tf_efficientnet_b0_a2.pth")
    elif model_arch == 'EfficientNet-b1-A2':
        timm_name = "tf_efficientnet_b1.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b1_a2_0-d342a7bf.pth", new_name="tf_efficientnet_b1_a2.pth")
        model = timm.create_model("tf_efficientnet_b1.in1k", wc.RSB_ROOT + "tf_efficientnet_b1_a2.pth")
    elif model_arch == 'EfficientNet-b2-A2':
        timm_name = "tf_efficientnet_b2.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b2_a2_0-ae4f4996.pth", new_name="tf_efficientnet_b2_a2.pth")
        model = timm.create_model("tf_efficientnet_b2.in1k", wc.RSB_ROOT + "tf_efficientnet_b2_a2.pth")
    elif model_arch == 'EfficientNet-b3-A2':
        timm_name = "tf_efficientnet_b3.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b3_a2_0-e183dbec.pth", new_name="tf_efficientnet_b3_a2.pth")
        model = timm.create_model("tf_efficientnet_b3.in1k", wc.RSB_ROOT + "tf_efficientnet_b3_a2.pth")
    elif model_arch == 'EfficientNet-b4-A2':
        timm_name = "tf_efficientnet_b4.in1k"
        download_file(url=wc.RSB_LINK + "tf_efficientnet_b4_a2_0-bc5f172e.pth", new_name="tf_efficientnet_b4_a2.pth")
        model = timm.create_model("tf_efficientnet_b4.in1k", wc.RSB_ROOT + "tf_efficientnet_b4_a2.pth")
    
    elif model_arch == 'EfficientNetv2-M-A2':
        timm_name = "efficientnetv2_rw_m.agc_in1k"
        download_file(url=wc.RSB_LINK + "efficientnetv2_rw_m_a2_0-12297cd3.pth", new_name="efficientnetv2_rw_m_a2.pth")
        model = timm.create_model("efficientnetv2_rw_m.agc_in1k", wc.RSB_ROOT + "efficientnetv2_rw_m_a2.pth")
    elif model_arch == 'EfficientNetv2-S-A2':
        timm_name = "efficientnetv2_rw_s.ra2_in1k"
        download_file(url=wc.RSB_LINK + "efficientnetv2_rw_s_a2_0-cafb8f99.pth", new_name="efficientnetv2_rw_s_a2.pth")
        model = timm.create_model("efficientnetv2_rw_s.ra2_in1k", wc.RSB_ROOT + "efficientnetv2_rw_s_a2.pth")
    

    elif model_arch == 'RegNety-040-A2':
        timm_name = "regnety_040.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_040_a2_0-acda0189.pth", new_name="regnety_040_a2.pth")
        model = timm.create_model("regnety_040.pycls_in1k", wc.RSB_ROOT + "regnety_040_a2.pth")
    elif model_arch == 'RegNety-080-A2':
        timm_name = "regnety_080.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_080_a2_0-2298ae4e.pth", new_name="regnety_080_a2.pth")
        model = timm.create_model("regnety_080.pycls_in1k", wc.RSB_ROOT + "regnety_080_a2.pth")
    elif model_arch == 'RegNety-160-A2':
        timm_name = "regnety_160.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_160_a2_0-6631355e.pth", new_name="regnety_160_a2.pth")
        model = timm.create_model("regnety_160.pycls_in1k", wc.RSB_ROOT + "regnety_160_a2.pth")
    elif model_arch == 'RegNety-320-A2':
        timm_name = "regnety_320.pycls_in1k"
        download_file(url=wc.RSB_LINK + "regnety_320_a2_0-a9fedcbf.pth", new_name="regnety_320_a2.pth")
        model = timm.create_model("regnety_320.pycls_in1k", wc.RSB_ROOT + "regnety_320_a2.pth")

    elif model_arch == 'ResNet18-A2':
        timm_name = "resnet18"
        download_file(url=wc.RSB_LINK + "resnet18_a2_0-b61bd467.pth", new_name="resnet18_a2.pth")
        model = timm.create_model("resnet18", wc.RSB_ROOT + "resnet18_a2.pth")
    elif model_arch == 'ResNet34-A2':
        timm_name = "resnet34"
        download_file(url=wc.RSB_LINK + "resnet34_a2_0-82d47d71.pth", new_name="resnet34_a2.pth")
        model = timm.create_model("resnet34", wc.RSB_ROOT + "resnet34_a2.pth")
    elif model_arch == 'ResNet50-A2':
        timm_name = "resnet50"
        download_file(url=wc.RSB_LINK + "resnet50_a2_0-a2746f79.pth", new_name="resnet50_a2.pth")
        model = timm.create_model("resnet50", wc.RSB_ROOT + "resnet50_a2.pth")
    elif model_arch == 'ResNet50d-A2':
        timm_name = "resnet50d.gluon_in1k"
        download_file(url=wc.RSB_LINK + "resnet50d_a2_0-a3adc64d.pth", new_name="resnet50d_a2.pth")
        model = timm.create_model("resnet50d.gluon_in1k", wc.RSB_ROOT + "resnet50d_a2.pth")
    elif model_arch == 'ResNet101-A2':
        timm_name = "resnet101"
        download_file(url=wc.RSB_LINK + "resnet101_a2_0-6edb36c7.pth", new_name="resnet101_a2.pth")
        model = timm.create_model("resnet101", wc.RSB_ROOT + "resnet101_a2.pth")
    elif model_arch == 'ResNet152-A2':
        timm_name = "resnet152"
        download_file(url=wc.RSB_LINK + "resnet152_a2_0-b4c6978f.pth", new_name="resnet152_a2.pth")
        model = timm.create_model("resnet152", wc.RSB_ROOT + "resnet152_a2.pth")

    elif model_arch == 'ResNext50-32x4d-A2':
        timm_name = "resnext50_32x4d.gluon_in1k"
        download_file(url=wc.RSB_LINK + "resnext50_32x4d_a2_0-efc76add.pth", new_name="resnext50_32x4d_a2.pth")
        model = timm.create_model("resnext50_32x4d.gluon_in1k", wc.RSB_ROOT + "resnext50_32x4d_a2.pth")
    elif model_arch == 'SeNet154-A2':
        timm_name = "senet154.gluon_in1k"
        download_file(url=wc.RSB_LINK + "gluon_senet154_a2_0-63cb3b08.pth", new_name="gluon_senet154_a2.pth")
        model = timm.create_model("senet154.gluon_in1k", wc.RSB_ROOT + "gluon_senet154_a2.pth")
    
    elif model_arch == 'SeNet154':
        timm_name = "senet154.gluon_in1k"
    elif model_arch == 'ResNet50d':
        timm_name = "resnet50d.gluon_in1k"
    elif model_arch == 'RegNet-y-4gf':
        timm_name = "regnety_040.pycls_in1k"

    ### DINOv2 Register Tokens
    elif model_arch == "ViT-b-14-dinov2-reg":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
    elif model_arch == "ViT-s-14-dinov2-reg":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
    elif model_arch == "ViT-l-14-dinov2-reg":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc')
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)

    #Fine-Tuned CLIP 
    elif "CLIP-B32-V-OpenAI" == model_arch:
        timm_name = "vit_base_patch32_clip_224.openai_ft_in1k"
    elif "CLIP-B32-V-Laion2B" == model_arch:
        timm_name = "vit_base_patch32_clip_224.laion2b_ft_in1k"
    elif "CLIP-B16-V-OpenAI" == model_arch:
        timm_name = "vit_base_patch16_clip_224.openai_ft_in1k"
    elif "CLIP-B16-V-Laion2B" == model_arch:
        timm_name = "vit_base_patch16_clip_224.laion2b_ft_in1k"
    
    #Fine-Tuned DINOv2
    elif "ViT-s-14-dinoV2-FT" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
        ckpt = torch.load(wc.DINOV2_S_FT)["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
    elif "ViT-b-14-dinoV2-FT" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
        ckpt = torch.load(wc.DINOV2_B_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
    elif "ViT-l-14-dinoV2-FT" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_lc')
        ckpt = torch.load(wc.DINOV2_L_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)

    elif "ViT-l-14-dinoV2-FT-Reg" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg_lc')
        ckpt = torch.load(wc.DINOV2_L_REG_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
    elif "ViT-b-14-dinoV2-FT-Reg" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg_lc')
        ckpt = torch.load(wc.DINOV2_B_REG_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)
    elif "ViT-s-14-dinoV2-FT-Reg" == model_arch:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg_lc')
        ckpt = torch.load(wc.DINOV2_S_REG_FT, map_location='cpu')["model"]
        model.load_state_dict(ckpt)
        transforms_list = [
            tv_transforms.Resize(256, interpolation=tv_transforms.InterpolationMode.BICUBIC),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        transform = tv_transforms.Compose(transforms_list)

    #EfficientNet http://proceedings.mlr.press/v97/tan19a.html?ref=jina-ai-gmbh.ghost.io
    #https://github.com/pytorch/vision/issues/7744
    def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)
    WeightsEnum.get_state_dict = get_state_dict
    
    if "EfficientNet-B0" == model_arch:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()
    elif "EfficientNet-B1" == model_arch:
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        transform = models.EfficientNet_B1_Weights.IMAGENET1K_V1.transforms()
    elif "EfficientNet-B2" == model_arch:
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        transform = models.EfficientNet_B2_Weights.IMAGENET1K_V1.transforms()
    elif "EfficientNet-B3" == model_arch:
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        transform = models.EfficientNet_B3_Weights.IMAGENET1K_V1.transforms()
    elif "EfficientNet-B4" == model_arch:
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        transform = models.EfficientNet_B4_Weights.IMAGENET1K_V1.transforms()
    elif "EfficientNet-B5" == model_arch:
        model = models.efficientnet_b5(weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1)
        transform = models.EfficientNet_B5_Weights.IMAGENET1K_V1.transforms()
    elif "EfficientNet-B6" == model_arch:
        model = models.efficientnet_b6(weights=models.EfficientNet_B6_Weights.IMAGENET1K_V1)
        transform = models.EfficientNet_B6_Weights.IMAGENET1K_V1.transforms()
    elif "EfficientNet-B7" == model_arch:
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        transform = models.EfficientNet_B7_Weights.IMAGENET1K_V1.transforms()
    
    model_name = model_name.replace("_", "-")

    if ("Salman2020" in model_name):
        transform = tv_transforms.Compose([
            tv_transforms.Resize(256),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
        ])

    if "Liu" in model_name or "Singh" in model_name:
        transform=tv_transforms.Compose([tv_transforms.Resize(256, tv_transforms.InterpolationMode.BICUBIC), tv_transforms.CenterCrop(224), tv_transforms.ToTensor()])
    
    if "BagNet" in model_name:
        transform = tv_transforms.Compose([
            tv_transforms.Resize(224),                  
            tv_transforms.CenterCrop(224),                
            tv_transforms.ToTensor(),                      
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
    
    if "Hiera" in model_name:
        transform = tv_transforms.Compose([
            tv_transforms.Resize(224),                  
            tv_transforms.CenterCrop(224),                
            tv_transforms.ToTensor(),                      
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

    if "Hiera-B-LP" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_base_224", pretrained=True, checkpoint="mae_in1k")
        state_dict = torch.load(wc.HIERA_LP_B)["model"]
        model.load_state_dict(state_dict) 
        transform = tv_transforms.Compose([
            tv_transforms.Resize(256, interpolation=3),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    if "Hiera-S-LP" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_small_224", pretrained=True, checkpoint="mae_in1k")
        state_dict = torch.load(wc.HIERA_LP_S)["model"]
        model.load_state_dict(state_dict) 
        transform = tv_transforms.Compose([
            tv_transforms.Resize(256, interpolation=3),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        

    if "Hiera-T-LP" == model_arch:
        model = torch.hub.load("facebookresearch/hiera", model="hiera_tiny_224", pretrained=True, checkpoint="mae_in1k")
        state_dict = torch.load(wc.HIERA_LP_T)["model"]
        model.load_state_dict(state_dict) 
        transform = tv_transforms.Compose([
            tv_transforms.Resize(256, interpolation=3),
            tv_transforms.CenterCrop(224),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    if timm_name:
        if timm.is_model(timm_name) and not model:
            model = timm.create_model(timm_name, pretrained=True)
        if not transform:
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config) 
          

    if "bcos" in model_name:
        transform=tv_transforms.Compose([tv_transforms.Resize(256), tv_transforms.CenterCrop(224), tv_transforms.ToTensor(), bcos_transforms.AddInverse()])
        model = StandardModel(model=model, model_name=model_name, transform=transform)

    elif "mobileclip" in model_name:
        model = MobileCLIPModel(model_arch, args.device)
    elif model_name in wc.MODEL_CONFIGS.keys():
        model = OpenCLIPModel(model_name, args.device)
    elif "metaclip" in model_name or model_name in ["convnext-large-d-320-clip", "convnext-large-d-clip", "convnext-base-w-320-clip", 
                                                    "clip-datacomp-xl-l14", "clip-datacomp-xl-b16", "clip-datacomp-xl-b32",
                                                    "clip-laion2b-b16", "clip-laion2b-b32", "clip-laion2b-l14",
                                                    "clipa-datacomp-l-l14", "clip-datacomp-l-b16",
                                                    "clip-dfn2b-l14", "clip-dfn2b-b16", 
                                                    "clip-commonpool-b16", "clip-commonpool-basic-b16", "clip-commonpool-text-b16", "clip-commonpool-image-b16", "clip-commonpool-laion-b16", "clip-commonpool-clip-b16", "clip-datacomp-l-b16",
                                                    "clip-commonpool-xl-clip-l14", "clip-commonpool-xl-laion-l14", "clip-commonpool-xl-l14"]:
        model = OpenCLIPModel(model_name, args.device)
    elif "clip" in model_name:
        model = ClipModel(model_arch, args.device)
    elif "SigLIP2" in model_name:
        model = SigLIP2Model(model_name, device=args.device)
    elif model_arch == "siglip-b-16":
        from open_clip import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-B-16-SigLIP')
        tokenizer = get_tokenizer('hf-hub:timm/ViT-B-16-SigLIP')
        model = SigLIPModel(model_name="siglip-b-16", model=model, preprocess=preprocess, tokenizer=tokenizer, device=args.device)
    elif model_arch == "siglip-l-16":
        from open_clip import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained('hf-hub:timm/ViT-L-16-SigLIP-256')
        tokenizer = get_tokenizer('hf-hub:timm/ViT-L-16-SigLIP-256')
        model = SigLIPModel(model_name="siglip-l-16", model=model, preprocess=preprocess, tokenizer=tokenizer, device=args.device)
    elif model_arch in ["ViT-b-16-DINO-LP",  "ViT-s-16-DINO-LP"]:
        model = DINOModel(model, head, model_arch, transform)
    else:
        model = StandardModel(model=model, model_name=model_name, transform=transform)

 
    if model != None:
        print(model_name, "Loaded")
    else:
        raise ValueError("Model not implemented:", model_arch)

    return model


def download_dataset(dataset_name):
    """
    Downloads the dataset specified by dataset_name

    :dataset_name: Dataset to download. Currently supported: Background Challenge Data, ImageNet-9l, ImageNet-R, Modelvshuman
    """

    target_dir = con._DATA_DIR
    if dataset_name == "bg_challenge":
        download_url = "https://github.com/MadryLab/backgrounds_challenge/releases/download/data/backgrounds_challenge_data.tar.gz"
    elif dataset_name == "imagenet-r":
        download_url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    elif dataset_name == "blur":
        download_url = "https://zenodo.org/records/3565846/files/blur.tar?download=1"
    elif dataset_name == "digital":
        download_url = "https://zenodo.org/records/3565846/files/digital.tar?download=1"
    elif dataset_name == "noise":
        download_url = "https://zenodo.org/records/3565846/files/noise.tar?download=1"
    elif dataset_name == "weather":
        download_url = "https://zenodo.org/records/3565846/files/weather.tar?download=1"
    else:
        target_dir = con._MODEL_VS_HUMAN_DIR
        download_url = "https://github.com/bethgelab/model-vs-human/releases/download/v0.1/{NAME}.tar.gz"
        download_url = download_url.format(NAME=dataset_name)
    
    if os.path.exists(target_dir + "/" + dataset_name):
        return
    
    response = requests.get(download_url, stream=True)
    if response.status_code == 200:
        total_length = response.headers.get('content-length')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        dataset_file = join(target_dir, f'{dataset_name}.tar.gz')
        print(f"Downloading dataset {dataset_name} to {dataset_file}")
        with open(dataset_file, 'wb') as fd:
            if total_length is None:  
                fd.write(response.content)
            else:
                for chunk in tqdm(response.iter_content(chunk_size=4096)):
                    fd.write(chunk)
        shutil.unpack_archive(dataset_file, target_dir)
        os.remove(dataset_file)
        return True
    else:
        return False

def list_models():
    """
    Lists every implemented model and subgroups of the QUBA model Zoo
    """
    model_list = [
        'AlexNet', 'GoogLeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'VGG11-bn', 
        'VGG13-bn', 'VGG16-bn', 'VGG19-bn', 'ResNet18', 'ResNet34', 'ResNet50', 
        'ResNet101', 'ResNet152', 'WRN-50-2', 'WRN-101-2', 'SqueezeNet', 'InceptionV3', 
        'InceptionV4', 'Inception-ResNetv2', 'ResNeXt50-32x4d', 'ResNeXt101-32x8d', 
        'ResNeXt101-64x4d', 'DenseNet121', 'DenseNet161', 'DenseNet169', 'DenseNet201', 
        'Xception', 'MobileNetV2', 'ShuffleNet-v2-05', 'ShuffleNet-v2-1', 'ShuffleNet-v2-15', 
        'ShuffleNet-v2-2', 'NasNet-l', 'MobileNetV3-s', 'MobileNetV3-l', 'MobileNetV3-l-21k', 
        'BagNet9', 'BagNet17', 'BagNet33', 'MnasNet-05', 'MnasNet-075', 'MnasNet-1', 'MnasNet-13', 
        'EfficientNet-B0', 'EfficientNet-B1', 'EfficientNet-B2', 'EfficientNet-B3', 'EfficientNet-B4', 
        'EfficientNet-B5', 'EfficientNet-B6', 'EfficientNet-B7', 'NS-EfficientNet-B0', 
        'NS-EfficientNet-B1', 'NS-EfficientNet-B2', 'NS-EfficientNet-B3', 'NS-EfficientNet-B4', 
        'NS-EfficientNet-B5', 'NS-EfficientNet-B6', 'NS-EfficientNet-B7', 'Salman2020Do-RN50-2', 
        'Salman2020Do-RN50', 'BiTM-resnetv2-50x1', 'BiTM-resnetv2-50x3', 'BiTM-resnetv2-101x1', 
        'BiTM-resnetv2-152x2', 'RegNet-y-400mf', 'RegNet-y-800mf', 'RegNet-y-1-6gf', 'RegNet-y-3-2gf', 
        'RegNet-y-8gf', 'RegNet-y-16gf', 'RegNet-y-32gf', 'ViT-b-16', 'ViT-l-16', 'ViT-b-32', 'ViT-l-32', 
        'Swin-T', 'Swin-S', 'Swin-B', 'EfficientNet-v2-S', 'EfficientNet-v2-S-21k', 'EfficientNet-v2-M', 
        'EfficientNet-v2-M-21k', 'EfficientNet-v2-L', 'EfficientNet-v2-L-21k', 'DeiT-t', 'DeiT-s', 'DeiT-b', 
        'ConViT-t', 'ConViT-s', 'ConViT-b', 'CaiT-xxs24', 'CaiT-xs24', 'CaiT-s24', 'CrossViT-9dagger', 
        'CrossViT-15dagger', 'CrossViT-18dagger', 'XCiT-s24-16', 'XCiT-m24-16', 'XCiT-l24-16', 'LeViT-128', 
        'LeViT-256', 'LeViT-384', 'PiT-t', 'PiT-xs', 'PiT-s', 'PiT-b', 'CoaT-t-lite', 'CoaT-mi-lite', 
        'CoaT-s-lite', 'CoaT-me-lite', 'MaxViT-t', 'MaxViT-b', 'MaxViT-l', 'DeiT3-s', 'DeiT3-s-21k', 
        'DeiT3-m', 'DeiT3-m-21k', 'DeiT3-b', 'DeiT3-b-21k', 'DeiT3-l', 'DeiT3-l-21k', 'MViTv2-t', 
        'MViTv2-s', 'MViTv2-b', 'MViTv2-l', 'SwinV2-T-W8', 'SwinV2-S-W8', 'SwinV2-B-W8', 
        'SwinV2-t-W16', 'SwinV2-s-Win16', 'SwinV2-b-Win16', 'SwinV2-b-Win12to16-21k', 
        'SwinV2-l-Win12to16-21k', 'ViT-t5-16', 'ViT-t5-16-21k', 'ViT-t11-16', 'ViT-t11-16-21k', 
        'ViT-t21-16', 'ViT-t21-16-21k', 'ViT-s-16', 'ViT-s-16-21k', 'ViT-b-16-21k', 'ViT-b-32-21k', 
        'ViT-l-16-21k', 'ViT-l-32-21k', 'ConvNext-T', 'ConvNext-T-21k', 'ConvNext-S', 'ConvNext-S-21k', 
        'ConvNext-B', 'ConvNext-B-21k', 'ConvNext-L', 'ConvNext-L-21k', 'BeiT-b', 'EfficientFormer-l1', 
        'EfficientFormer-l3', 'EfficientFormer-l7', 'DaViT-t', 'DaViT-s', 'DaViT-b', 
        'Liu2023Comprehensive-Swin-B', 'Liu2023Comprehensive-Swin-L', 'Liu2023Comprehensive-ConvNeXt-B', 
        'Liu2023Comprehensive-ConvNeXt-L', 'Singh2023Revisiting-ConvNeXt-T-ConvStem', 
        'Singh2023Revisiting-ConvNeXt-S-ConvStem', 'Singh2023Revisiting-ConvNeXt-B-ConvStem', 
        'Singh2023Revisiting-ConvNeXt-L-ConvStem', 'Singh2023Revisiting-ViT-B-ConvStem', 'ConvNextV2-N', 
        'ConvNextV2-N-21k', 'ConvNextV2-T', 'ConvNextV2-T-21k', 'ConvNextV2-B', 'ConvNextV2-B-21k', 
        'ConvNextV2-L', 'ConvNextV2-L-21k', 'Hiera-T', 'Hiera-S', 'Hiera-B', 'Hiera-B-Plus', 'Hiera-L', 
        'EVA02-t-21k', 'EVA02-s-21k', 'EVA02-b-21k', 'InceptionNext-t', 'InceptionNext-s', 
        'InceptionNext-b', 'FastViT-sa12', 'FastViT-sa24', 'FastViT-sa36', 'BeiTV2-b', 'SeNet154', 
        'ResNet50d', 'ResNeXt50-32x4d-YFCCM100', 'ResNet50-yfcc100m', 'ResNet50-ig1B', 
        'ResNeXt101-32x8d-IG1B', 'ResNeXt50-32x4d-IG1B', 'ResNet18-IG1B', 'vit-b-16-mae-ft', 
        'vit-b-16-mae-lp', 'Hiera-B-LP', 'ViT-b-16-DINO-LP', 'ViTB-DINO-FT', 'ResNet50-DINO-LP', 
        'ResNet50-DINO-FT', 'ViT-l-14-dinoV2-LP', 'ViT-b-14-dinoV2-LP', 'ViT-s-14-dinoV2-LP', 
        'vit-t-16-21k', 'siglip-b-16', 'clip-resnet50', 'clip-vit-b-16', 'clip-resnet101', 
        'clip-vit-b-32', 'mobileclip-s0', 'mobileclip-s1', 'mobileclip-s2', 'mobileclip-b', 
        'EfficientNet-b0-A1', 'EfficientNet-b1-A1', 'EfficientNet-b2-A1', 'EfficientNet-b3-A1', 
        'EfficientNet-b4-A1', 'EfficientNetv2-M-A1', 'EfficientNetv2-S-A1', 'RegNety-040-A1', 
        'RegNety-080-A1', 'RegNety-160-A1', 'RegNety-320-A1', 'ResNet101-A1', 'ResNet152-A1', 
        'ResNet18-A1', 'ResNet34-A1', 'ResNet50-A1', 'ResNet50d-A1', 'ResNext50-32x4d-A1', 'SeNet154-A1',
        'EfficientNet-b0-A2', 'EfficientNet-b1-A2', 'EfficientNet-b2-A2', 'EfficientNet-b3-A2', 
        'EfficientNet-b4-A2', 'EfficientNetv2-M-A2', 'EfficientNetv2-S-A2', 'RegNety-040-A2', 
        'RegNety-080-A2', 'RegNety-160-A2', 'RegNety-320-A2', 'ResNet101-A2', 'ResNet152-A2', 
        'ResNet18-A2', 'ResNet34-A2', 'ResNet50-A2', 'ResNet50d-A2', 'ResNext50-32x4d-A2', 'SeNet154-A2', 
        'EfficientNet-b0-A3', 'EfficientNet-b1-A3', 'EfficientNet-b2-A3', 'EfficientNet-b3-A3', 
        'EfficientNet-b4-A3', 'EfficientNetv2-M-A3', 'EfficientNetv2-S-A3', 'RegNety-040-A3', 
        'RegNety-080-A3', 'RegNety-160-A3', 'RegNety-320-A3', 'ResNet101-A3', 'ResNet152-A3', 
        'ResNet18-A3', 'ResNet34-A3', 'ResNet50-A3', 'ResNet50d-A3', 'ResNext50-32x4d-A3', 'SeNet154-A3', 
        'bcos-convnext-base', 'bcos-convnext-tiny', 'bcos-DenseNet121', 'bcos-DenseNet161', 
        'bcos-DenseNet169', 'bcos-DenseNet201', 'bcos-ResNet152', 'bcos-ResNet18', 'bcos-ResNet34', 
        'bcos-ResNet50', 'bcos-simple-vit-b-patch16-224', 'RegNet-y-4gf', 'mobileclip-blt', 
        'ViT-s-16-DINO-LP', 'siglip-l-16', 'bcos-ResNet101', 'metaclip-b16', 'convnext-large-d-clip', 
        'metaclip-l14', 'Singh2023Revisiting-ViT-S-ConvStem', 'convnext-base-w-320-clip', 
        'convnext-large-d-320-clip', 'Hiera-S-LP', 'Hiera-T-LP',
    ]
    group_list = [key for key in con.MODEL_MAP]
    table = {"Models": model_list, "Groups": group_list}
    return table

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
