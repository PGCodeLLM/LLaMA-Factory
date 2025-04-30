from pymongo import MongoClient
import datasets
import json

MONGO_URI = "mongodb://bmc:GwvDjDyUnm1GpRT6sMAq7rUo44EDmzuv02Tn9n5mmqvZyn3Zvsee4ozdCGFN57qXRqKEYethBPQfErCGE4oAr3feVuqjpcBuF2em@lux-2-cyber-04:26969/"
DATABASE_NAME = "swe_gym_plus"
COLLECTION_NAME = "train"

_DESCRIPTION = ""
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""


client = MongoClient(MONGO_URI, connectTimeoutMS=100_000_000, serverSelectionTimeoutMS=100_000_000)
db = client[DATABASE_NAME]   
collection = db[COLLECTION_NAME]

class SweGym(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features(
            {
                'text': datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION, features=features, homepage=_HOMEPAGE, license=_LICENSE, citation=_CITATION
        )
    
    def _split_generators(self, dl_manager: datasets.DownloadManager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"query": ""}),
        ]
    
    def _generate_examples(self, query):
        cursor = collection.find({
            'problem_statement': {'$exists': True},
            'problem_statement': {'$ne': ''},
        },
        {
            'instance_id': 1,
            'problem_statement': 1,
            'patch': 1,
            'test_patch': 1,
        })
        for row in cursor:
            instance_id = row['instance_id']
            problem_statement = row['problem_statement']
            patch = row['patch']
            test_patch = row['test_patch']
            text = f"Problem statement:\n{problem_statement}\nPatch:\n{patch}\nTest patch:\n{test_patch}"
            yield instance_id, {'text': text}