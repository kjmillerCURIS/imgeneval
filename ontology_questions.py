import os
import sys
import copy
from collections import Counter
from PIL import Image
import json
import numpy as np
import openai
from openai_utils import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
import re
import shutil
import tiktoken
from influence_on_human_ratings import load_human_ratings_data, adjust_impath
from ontology_toolset import setup_models, run_object_detector, check_scene_description, check_semantic_relationship, check_predefined_spatial_relationship, check_attribute_comparison_relationship, check_attribute, make_bbox_description
from ontology_match_generator import match_generator
from ontology_logger import Logger


PRINT_NUM_TOKENS = 0

#generate this many graphs, then pick the one with the median colon-count
#this is a quick-n-dirty way to do a "voting" scheme
#(assume there are only two possible answers. Median will pick the more frequent one.)
NUM_FOR_MEDIAN_LENGTH = 3

MY_INPUTS = [
        'A painting where the mountain is depicted as taller than the trees in the foreground.',
        'A bustling city street with cars, but no bicycles.',
        'After a snowfall, a group of kids builds a fort with blocks of snow piled up.',
        'Two cats with striped tails sitting side by side, one with a bow tie and the other without.',
        'A plastic bag is placed on the chair, but it contains nothing.',
        'A rabbit standing on a stump looks more nervous than another rabbit not on a stump.',
        'On display, a long, white dress contrasts sharply with a short, dark dress beside it.',
        'A playful cat not batting at a dangling string but chasing a laser pointer.',
        'A cyclist racing down a winding mountain path.',
        'a person without a hat pushes a person with a hat sitting in a box.',
        'Five ants are carrying biscuits, and an ant that is not carrying biscuits is standing on a green leaf directing them.',
        'Five canvas bags sit to the right of three woolen hats.',
        'A team of sheep and a team of farmers are having a tug-of-war, and there are more sheep than farmers.',
        'A girl has three flower decorations in her hair, all of them daisies.']
#        'an old person kisses a young person',
#        'a young person kisses an old person',
#        'the taller person hugs the shorter person',
#        'the shorter person hugs the taller person',
#        'the masked wrestler hits the unmasked wrestler',
#        'the unmasked wrestler hits the masked wrestler',
#        'a person watches an animal',
#        'an animal watches a person',
#        'the person without earrings pays the person with earrings',
#        'the person with earrings pays the person without earrings']

TASK_DESCRIPTION = '''
You are an expert in semantic graph generation. Your task is to analyze the provided text and generate a detailed semantic graph in JSON format. The semantic graph should include:
1. **Objects**: A list of objects mentioned in the text, each with the following attributes:
- 'object_id': A unique identifier for the object.
- 'name': The name of the object.
- 'is_group': "Yes" if this node represents an indeterminate number of objects, "No" if it is only one object. For example, "a group of cats", "a murder of crows", "some avocados" would all be "Yes", whereas "cat", "crow", "avocado" would be "No". Do NOT make a group if the number of objects is specified! For example, for "three cats" you should make "three" nodes that are each "cat" and singular.
- 'attributes': A list of attributes describing the object (e.g., color, size, material, body markings, clothing). This could be an empty list if no attributes are mentioned. Each attribute should have:
– 'attribute_name': 1-5 words describing the attribute
– 'is_negation': “Yes” if it has a negation word (e.g. “not red”, “without a tie”, "not wearing earrings", "not on fire"), “No” if it doesn’t (e.g. “red”, “with a tie”, "wearing earrings", "on fire")
2. **Relationships**: A list of relationships between objects, each with the following attributes:
- 'subject_ids': A list containing the 'object_id' of each subject in the relationship. There must be at least one subject!
- 'object_ids': A list containing the 'object_id' of each object in the relationship. There must be at least one object!
- 'type': Possible types are:
-- 'attribute comparison': comparing attributes, e.g. “bigger than”, “brighter than”
-- 'spatial only': relationship is fully described by “next to”, “above”, “below”, “left of”, “right of”, “in front”, “behind”, “on”, “in”, “near”
– 'spatial semantic': relationship has a spatial layout, but isn't 'spatial only', e.g. “lying on”, “chasing”, “fighting”
– 'negative spatial only': relationship can be described as the negation of a 'spatial only', e.g. “not next to”, "not in"
– 'negative spatial semantic': relationship can be described as the negation of a 'spatial semantic', e.g. “not lying on”, “not chasing”, “not fighting”
-- 'count comparison': comparing the cardinality of two groups, e.g. "more apples than oranges", "less flies than mosquitoes", "fewer fish than deer". Both subject and object must have "is_group" set to "Yes".
- 'predicate': 1-5 words describing the relationship
3. **Scene**: A single item containing global information about the scene in the image:
- 'description': 1-3 words describing the scene or style, e.g. "urban", "rural", "auditorium", "painting", "playful", "amusement park", etc., or "N/A" if it is not specified.
- 'absent_object_names': list of names of objects explicitly specified as being absent from the image. Leave it empty if no objects are specified as being absent.
'''

ICL_PART = '''
Here are examples:
Example text: "three children eating cookies, and a child not eating any"
Example graph: {"objects": [ {"object_id": 1, "name": "child", "is_group": "No", "attributes": []}, {"object_id": 2, "name": "child", "is_group": "No", "attributes": []}, {"object_id": 3, "name": "child", "is_group": "No", "attributes": []}, {"object_id": 4, "name": "child", "is_group": "No", "attributes": []}, {"object_id": 5, "name": "cookie", "is_group": "Yes", "attributes": []} ], "relationships": [ {"subject_ids": [1,2,3], "object_ids": [5], "type": "spatial semantic", "predicate": "eating"}, {"subject_ids": [4], "object_ids": [5], "type": "negative spatial semantic", "predicate": "not eating"} ], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: "a sketch of two succulent potatoes on top of each other"
Example graph: {"objects": [ {"object_id": 1, "name": "potato", "is_group" : "No", "attributes": [{"attribute_name": "succulent", "is_negation": "No"}]}, {"object_id" : 2, "name" : "potato", "is_group" : "No", "attributes": [{"attribute_name" : "succulent", "is_negation": "No"}]} ], "relationships": [ {"subject_ids": [1], "object_ids": [2], "type" : "spatial only", "predicate" : "on top of"}], "scene": {"description": "sketch", "absent_object_names": []}}
Example text: “The zebra who is not lying on the table is redder than the panda without a tie who is biting the table”
Example graph: {"objects": [ {"object_id": 1, "name": "zebra", "is_group" : "No", "attributes": [ ] }, { "object_id": 2, "name": "panda", "is_group" : "No", "attributes": [{“attribute_name”: “without a tie”, “is_negation”: “Yes” }] }, { "object_id": 3, "name": "table", "is_group" : "No", "attributes": [] }], "relationships": [ { "subject_ids": [1], "object_ids": [3], "type": "negative spatial semantic", "predicate": "not lying on" }, { "subject_ids": [2], "object_ids": [3], "type": "spatial semantic", "predicate": "biting"},  { "subject_ids": [1], "object_ids": [2], "type": “attribute comparison", "predicate": "redder than"}], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: "there are more orange penguins than spotted zebras at the zoo, and no monkeys"
Example graph: {"objects": [ {"object_id": 1, "name": "penguin", "is_group": "Yes", "attributes": [{"attribute_name": "orange", "is_negation": "No"}]}, {"object_id": 2, "name": "zebra", "is_group": "Yes", "attributes": [{"attribute_name": "spotted", "is_negation": "No"}]}], "relationships": [ {"subject_ids": [1], "object_ids": [2], "type": "count comparison", "predicate": "more"}], "scene"{"description": "zoo", "absent_object_names": ["monkey"]}}
Example text: "a thinner squirrel and a fatter one"
Example graph: {"objects": [ {"object_id": 1, "name": "squirrel", "is_group": "No"}, {"object_id": 2, "name": "squirrel", "is_group": "No"} ], "relationships":[{"subject_ids": [1], "object_ids": [2], "type": "attribute comparison", "predicate": "thinner"}], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: "a thinner squirrel chasing a fatter one"
Example graph: {"objects": [ {"object_id": 1, "name": "squirrel", "is_group": "No"}, {"object_id": 2, "name": "squirrel", "is_group": "No"} ], "relationships":[{"subject_ids": [1], "object_ids": [2], "type": "attribute comparison", "predicate": "thinner"}, {"subject_ids": [1], "object_ids": [2], "type": "spatial semantic", "predicate": "chasing"}], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: “an orange chasing an apple, and no grapes or pears”
Example graph: {“objects”: [ {“object_id”: 1, “name”: “apple”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “orange”, "is_group" : "No", “attributes”:[]} ], “relationships”:[{“subject_ids” : [1], “object_ids” : [2], “type” : “spatial semantic”, “predicate” : “chasing”}], "scene": {"description": "N/A", "absent_object_names": ["grape", "pear"]}}
Example text: “a chinchilla next to an umbrella at the beach”
Example graph: {“objects”: [ {“object_id”: 1, “name”: “chinchilla”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “umbrella”, "is_group" : "No", “attributes”:[]} ], “relationships”:[{“subject_ids” : [1], “object_ids” : [2], “type” : “spatial only”, “predicate” : “next to”}], "scene": {"description": "beach", "absent_object_names": []}}
Example text: “a painting of a mouse lying on a yellow table”
Example graph: {“objects”: [ {“object_id”: 1, “name”: “mouse”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “table”, "is_group" : "No", “attributes”:[{“attribute_name” : “yellow”, “is_negation”:”No”}]} ], “relationships”:[{“subject_ids” : [1], “object_ids” : [2], “type” : “semantic spatial”, “predicate” : “lying on”}], "scene": {"description": "painting", "absent_object_names": []}}
Example text: "a man in a house and a woman not in a house"
Example graph: {"objects": [ {"object_id": 1, "name": "man", "is_group" : "No", "attributes":[]}, {"object_id": 2, "name": "woman", "is_group" : "No", "attributes":[]}, {"object_id" : 3, "name": "house", "is_group" : "No", "attributes":[]} ], "relationships" : [ {"subject_ids" : [1], "object_ids": [3], "type": "spatial", "predicate": "in"}, {"subject_ids": [2], "object_ids": [3], "type": "negative spatial only", "predicate": "not in"} ], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: “a fish on top and a fish on bottom”
Example graph: {“objects”: [ {“object_id”: 1, “name”: “fish”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “fish”, "is_group" : "No", "attributes":[]} ], “relationships”:[{“subject_ids” : [1], “object_ids” : [2], “type” : “spatial only”, “predicate” : "above"}], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: "a mosaic of an orange peanut with a monocle"
Example graph: {"objects": [ {"object_id": 1, "name": "peanut", "is_group" : "No", "attributes":[{"attribute_name" : "orange", "is_negation" : "No"}, {"attribute_name" : "with a monocle", "is_negation" : "No"}]} ], "relationships" : [], "scene": {"description": "mosaic", "absent_object_names": []}}
Example text: "a mosaic of an orange peanut without a monocle"
Example graph: {"objects": [ {"object_id": 1, "name": "peanut", "is_group" : "No", "attributes":[{"attribute_name" : "orange", "is_negation" : "No"}, {"attribute_name" : "without a monocle", "is_negation" : "Yes"}]} ], "relationships" : [], "scene": {"description": "mosaic", "absent_object_names": []}}
Example text: “a still life with three oranges in front of four pears, and no cheese”
Example graph: {“objects”: [ {“object_id”: 1, “name”: "orange”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “orange”, "is_group" : "No", "attributes":[]}, {“object_id”: 3, “name”: “orange”, "is_group": "No", "attributes":[]}, {“object_id”: 4, “name”: “pear”, "is_group": "No", "attributes":[]}, {“object_id”: 5, “name”: “pear”, "is_group" : "No", "attributes":[]}, {“object_id”: 6, “name”: “pear”, "is_group": "No", "attributes":[]}, {“object_id”: 7, “name”: “pear”, "is_group" : "No", "attributes":[]}], “relationships”:[{“subject_ids” : [1,2,3], “object_ids” : [4,5,6,7], “type” : “spatial only”, “predicate” : "in front of"}], "scene": {"description": "still life", "absent_object_names": ["cheese"]}}
Example text: "some purple pigeons and some violet wigeons"
Example graph: {"objects": [ {"object_id": 1, "name": "pigeon", "is_group": "Yes", "attributes":[{"attribute_name":"purple", "is_negation":"No"}]}, {"object_id": 2, "name": "wigeon", "is_group": "Yes", "attributes":[{"attribute_name":"violet", "is_negation":"No"}]} ], "relationships": [], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: "a murder of crows chasing a gaggle of geese"
Example graph: {"objects": [ {"object_id": 1, "name": "crow", "is_group": "Yes", "attributes":[]}, {"object_id": 2, "name": "goose", "is_group": "Yes", "attributes":[], "relationships": [{"subject_ids": [1], "object_ids": [2], "type" : "spatial semantic", "predicate" : "chasing"}], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: "a murder of crows chasing a boy"
Example graph: {"objects": [ {"object_id": 1, "name": "crow", "is_group": "Yes", "attributes":[]}, {"object_id": 2, "name": "boy", "is_group": "No", "attributes":[], "relationships": [{"subject_ids": [1], "object_ids": [2], "type" : "spatial semantic", "predicate" : "chasing"}], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: "a boy chasing a murder of crows"
Example graph: {"objects": [ {"object_id": 1, "name": "boy", "is_group": "No", "attributes":[]}, {"object_id": 2, "name": "crow", "is_group": "Yes", "attributes":[], "relationships": [{"subject_ids": [1], "object_ids": [2], "type" : "spatial semantic", "predicate" : "chasing"}], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: "a smaller murder of crows chasing a bigger gaggle of geese"
Example graph: {"objects": [ {"object_id": 1, "name": "crow", "is_group": "Yes", "attributes":[]}, {"object_id": 2, "name": "goose", "is_group": "Yes", "attributes":[], "relationships": [{"subject_ids": [1], "object_ids": [2], "type" : "spatial semantic", "predicate" : "chasing"}, {"subject_ids": [1], "object_ids": [2], "type": "count comparison", "predicate": "smaller"}], "scene": {"description": "N/A", "absent_object_names": []}}
Example text: "a murder of crows chasing a gaggle of geese, with more crows than geese"
Example graph: {"objects": [ {"object_id": 1, "name": "crow", "is_group": "Yes", "attributes":[]}, {"object_id": 2, "name": "goose", "is_group": "Yes", "attributes":[], "relationships": [{"subject_ids": [1], "object_ids": [2], "type" : "spatial semantic", "predicate" : "chasing"}, {"subject_ids": [1], "object_ids": [2], "type": "count comparison", "predicate": "more"}], "scene": {"description": "N/A", "absent_object_names": []}}
'''

INPUT_TEMPLATE = 'Here is the text: "%s"'
REDUNDANT_ATTRIBUTE_PROMPT = 'Is there any overlap between the attributes and the relationships? If so, please identify and remove the offending attributes from the graph.'
REDUNDANT_EDGE_PROMPT = 'Are there any edge pairs (i,j) and (j,i) that could be replaced with one edge (i,j)? If so, please do that.'

SPATIAL_VOCAB_ENTRIES = ['next to', 'on', 'in', 'above', 'below', 'to the right of', 'to the left of', 'near', 'in front of', 'behind']
SPATIAL_NONANSWER_ENTRY = 'i have no idea where they are!'


def process_reply(reply):
    start = reply.find('{')
    if start == -1:
        return None

    end = reply.rfind('}')
    if end == -1:
        return None

    return reply[start:end+1]


def generate_semantic_graph(my_input):
    
    # Initialize the conversation
    messages = [
        {"role": "system", "content": TASK_DESCRIPTION + '\n\n' + ICL_PART + '\n\n' + INPUT_TEMPLATE % my_input}
    ]
    
    encoding = tiktoken.encoding_for_model('gpt-4o')
    if PRINT_NUM_TOKENS:
        logger.log('Initial graph-generation prompt has %d tokens'%(len(encoding.encode(messages[0]['content']))))

    # Send the API request
    response = openai.chat.completions.create(
        model="gpt-4o",  # Use the GPT-4 model
        messages=messages
    )
    
    # Get the assistant's reply
    assistant_replyA = response.choices[0].message.content
    
    # Add the assistant's reply to the conversation history
    messages.append({"role": "assistant", "content": assistant_replyA})
    
    # User's follow-up question
    messages.append({"role": "user", "content": REDUNDANT_ATTRIBUTE_PROMPT})
    
    # Send the API request again
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    # Get the assistant's reply to the follow-up
    assistant_replyB = response.choices[0].message.content
    
    # Add the assistant's reply to the conversation history
    messages.append({"role": "assistant", "content": assistant_replyB})
    
    # User's follow-up question
    messages.append({"role": "user", "content": REDUNDANT_EDGE_PROMPT})
    
    if PRINT_NUM_TOKENS:
        logger.log('Final graph-generation messages have %d tokens total'%(sum([len(encoding.encode(msg['content'])) for msg in messages])))
    
    # Send the API request again
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    # Get the assistant's reply to the follow-up
    assistant_replyC = response.choices[0].message.content

    semantic_graph = process_reply(assistant_replyA)
    graphB = process_reply(assistant_replyB)
    if graphB:
        semantic_graph = graphB
    
    graphC = process_reply(assistant_replyC)
    if graphC:
        semantic_graph = graphC

    return semantic_graph, (assistant_replyA, assistant_replyB, assistant_replyC)


def process_yesno(yesno):
    return {'Yes' : True, 'No' : False}[yesno]


def parse_relationship(relationship):
    assert('subject_ids' in relationship)
    assert('object_ids' in relationship)
    assert('predicate' in relationship)
    assert('type' in relationship)
    assert(relationship['type'] in ['spatial only', 'spatial semantic', 'negative spatial only', 'negative spatial semantic', 'attribute comparison', 'count comparison'])
    return relationship


def parse_attribute(attribute):
    assert('attribute_name' in attribute)
    assert('is_negation' in attribute)
    return {'attribute_name' : attribute['attribute_name'], 'is_negation' : process_yesno(attribute['is_negation'])}


def parse_object(my_object):
    assert('object_id' in my_object)
    assert('name' in my_object)
    assert('is_group' in my_object)
    assert('attributes' in my_object)
    my_object['is_group'] = process_yesno(my_object['is_group'])
    my_object['attributes'] = [parse_attribute(attribute) for attribute in my_object['attributes']]
    return my_object


def parse_semantic_graph(semantic_graph):
    semantic_graph = json.loads(semantic_graph)
    object_list, relationships, scene = semantic_graph['objects'], semantic_graph['relationships'], semantic_graph['scene']
    object_list = [parse_object(my_object) for my_object in object_list]
    objects = {}
    for my_object in object_list:
        assert(my_object['object_id'] not in objects)
        objects[my_object['object_id']] = my_object

    relationships = [parse_relationship(relationship) for relationship in relationships]
    assert('description' in scene)
    assert('absent_object_names' in scene)
    assert(isinstance(scene['absent_object_names'], list))
    return objects, relationships, scene


def handle_negation(text, caches):
    message = {'role' : 'system', 'content' : 'You are an expert in negation. Please take the negated phrase "%s" and unnegate it. Put your answer by itself in the last line of output.'%text}
    if message['content'] in caches['LLM_cache']:
        reply = caches['LLM_cache'][message['content']]
    else:
        response = openai.chat.completions.create(model='gpt-4o', messages=[message])
        reply = response.choices[0].message.content
        caches['LLM_cache'][message['content']] = reply
    
    reply = reply.strip().split('\n')[-1].strip().strip('"').strip().rstrip('.')
    return reply


#project onto predefined spatial vocab
def handle_spatial_only_relationship(predicate, caches):
    options = []
    for spatial_vocab_entry in SPATIAL_VOCAB_ENTRIES:
        options.append('X ' + spatial_vocab_entry + ' Y')
        options.append('Y ' + spatial_vocab_entry + ' X')

    message = {'role' : 'system', 'content' : 'You are an expert in spatial relationships between objects. Your task is to map the spatial relationship "X %s Y" into a fixed vocabulary. Which of the following is closest to it in meaning: '%(predicate) + ', '.join(['"' + option + '"' for option in options]) + '? You must choose one of these options. Please put your answer by itself in the last line of your output.'}
    if message['content'] in caches['LLM_cache']:
        reply = caches['LLM_cache'][message['content']]
    else:
        response = openai.chat.completions.create(model='gpt-4o', messages=[message])
        reply = response.choices[0].message.content
        caches['LLM_cache'][message['content']] = reply

    reply = reply.strip().split('\n')[-1].strip().strip('"').strip().rstrip('.')
    if reply not in options:
        options_in_reply = [option for option in options if option in reply]
        assert(len(options_in_reply) == 1)
        reply = options_in_reply[0]

    assert(reply in options)
    flip = reply.startswith('Y')
    a, b = len('X '), len(' Y')
    if flip:
        a, b = b, a
    spatial_vocab_entry = reply[a:-b]
    return spatial_vocab_entry, flip


#extract spatial in terms of predefined spatial vocab (or say that it's not spatial)
def extract_spatial_from_spatial_semantic_relationship(predicate, subject_name, object_name, caches, logger):
    subject_name, object_name = subject_name.lower(), object_name.lower()
    if subject_name == object_name:
        subject_name, object_name = 'first ' + subject_name, 'second ' + object_name

    options = []
    for spatial_vocab_entry in SPATIAL_VOCAB_ENTRIES:
        options.append(subject_name + ' ' + spatial_vocab_entry + ' ' + object_name)
        options.append(object_name + ' ' + spatial_vocab_entry + ' ' + subject_name)

    options.append(SPATIAL_NONANSWER_ENTRY)
    message = {'role' : 'system', 'content' : 'You are an expert on describing relationships between objects in spatial terms. Your task is to describe "%s %s %s", in which the relationship is "%s" and the objects are "%s" and "%s". Which of the following would be the most appropriate description of their spatial relationship: '%(subject_name, predicate, object_name, predicate, subject_name, object_name) + ', '.join(['"' + option + '"' for option in options]) + '? You must choose one of these options. Please put your answer by itself in the last line of your output.'}
    if message['content'] in caches['LLM_cache']:
        reply = caches['LLM_cache'][message['content']]
    else:
        response = openai.chat.completions.create(model='gpt-4o', messages=[message])
        reply = response.choices[0].message.content
        caches['LLM_cache'][message['content']] = reply

    reply = reply.strip().split('\n')[-1].strip().strip('"').strip().rstrip('.').lower()
    if reply not in options:
        options_in_reply = [option for option in options if option in reply]
        if len(options_in_reply) != 1:
            logger.log('(could not extract spatial relationship from "%s %s %s", skipping spatial check (reply last line "%s"))'%(subject_name, predicate, object_name, reply))
            return 'N/A', False

        reply = options_in_reply[0]

    assert(reply in options)
    if reply == SPATIAL_NONANSWER_ENTRY:
        logger.log('(could not extract spatial relationship from "%s %s %s", skipping spatial check)'%(subject_name, predicate, object_name))
        return 'N/A', False
    else:
        flip = reply.startswith(object_name)
        a, b = len(subject_name + ' '), len(' ' + object_name)
        if flip:
            a, b = b, a
        spatial_vocab_entry = reply[a:-b]
        return spatial_vocab_entry, flip


#extract attribute so they can be compared
def extract_attribute_from_attribute_comparison_relationship(predicate, caches, logger):
    message = {"role": "system", "content": 'You are an expert in attribute comparison grammar. Please express the phrase "X %s Y" in the form "X is more ADJ than Y" or "X is less ADJ than Y", where "ADJ" is an adjective. Please put your answer by itself in the last line of your output. Change as little as possible about the original phrase.' % predicate}
    if message['content'] in caches['LLM_cache']:
        reply = caches['LLM_cache'][message['content']]
    else:
        response = openai.chat.completions.create(model='gpt-4o', messages=[message])
        reply = response.choices[0].message.content
        caches['LLM_cache'][message['content']] = reply

    reply = reply.strip().split('\n')[-1].strip().strip('"').strip().rstrip('.')
    if 'ADJ' in reply or not ((reply.startswith('X is more ') or reply.startswith('X is less ')) and reply.endswith(' than Y')):
        logger.log('(could not extract attribute from comparison "X %s Y", skipping attribute comparison check (LLM reply last line was "%s")'%(predicate, reply))
        return 'N/A', 'N/A'

    if reply.startswith('X is more '):
        return reply[len('X is more '):-len(' than Y')], '>'
    elif reply.startswith('X is less '):
        return reply[len('X is less '):-len(' than Y')], '<'
    else:
        assert(False)


#save some LLM calls by handling easy cases
#return None if we can't handle it
def handle_count_comparison_try_easy_cases(predicate):
    could_be_greater = (predicate in ['greater', 'more', 'bigger', 'larger'])
    could_be_less = (predicate in ['less', 'fewer', 'smaller', 'tinier', 'lesser'])
    assert(not (could_be_greater and could_be_less))
    if could_be_greater or could_be_less:
        return {True: '>', False: '<'}[could_be_greater]
    else:
        return None


#return ">" or "<" (or "N/A" if it can't figure it out, in which case the count-comparison is skipped)
def handle_count_comparison(predicate, caches, logger):
    try_crocodile = handle_count_comparison_try_easy_cases(predicate)
    if try_crocodile is not None:
        return try_crocodile

    message = {"role": "system", "content": 'You are an expert in count comparison grammar. Please decide if the predicate "%s" is best described by the mathematical symbol ">" or "<". You must choose one or the other. Please put your answer by itself (">" or "<") in the last line of your output.' % predicate}
    if message['content'] in caches['LLM_cache']:
        reply = caches['LLM_cache'][message['content']]
    else:
        response = openai.chat.completions.create(model='gpt-4o', messages=[message])
        reply = response.choices[0].message.content
        caches['LLM_cache'][message['content']] = reply

    reply = reply.strip().split('\n')[-1].strip().strip('"').strip().rstrip('.')
    could_be_greater = ('>' in reply)
    could_be_less = ('<' in reply)
    if could_be_greater and not could_be_less:
        return '>'
    elif could_be_less and not could_be_greater:
        return '<'
    else:
        logger.log('(could not resolve count comparison predicate "%s", skipping count comparison check (LLM reply last line was "%s"))'%(predicate, reply))
        return 'N/A'


def stringify_detection(my_object):
    if my_object['is_group']:
        return 'detection_[%s,:]' % str(my_object['object_id'])
    else:
        return 'detection_%s' % str(my_object['object_id'])


def do_count_checks(objects, detections):
    result = []
    group_names = [objects[k]['name'] for k in sorted(objects.keys()) if objects[k]['is_group']]
    cnt = Counter([objects[k]['name'] for k in sorted(objects.keys())])
    for name in sorted(cnt.keys()):
        if name in group_names:
            result.append(('Check that there are at least 2 %ss detected'%(name), int(len(detections[name]) >= 2)))
            continue

        result.append(('Check that there are %d %ss detected'%(cnt[name], name), int(len(detections[name]) == cnt[name])))

    return result


def check_object(image_pil, my_object, detection, models, caches):
    result = []

    #FIXME: consider uncommenting this if you see the matcher "cheating" by underpopulating some groups
    #I don't think this can happen unless there are multiple groups with the same name
    #And even then, I'm not sure if it would successfully "cheat"
    #But I'm keeping this code here just in case it comes up
    #if my_object['is_group']:
    #    result.append(('Check that %s{%s} group is matched to at least 1 detection'%(my_object['name'], str(my_object['object_id'])), int(len(detection) >= 1)))
    #    result.append(('Check that %s{%s} group is matched to at least 2 detections'%(my_object['name'], str(my_object['object_id'])), int(len(detection) >= 2)))

    for attribute in my_object['attributes']:
        attribute_name = attribute['attribute_name']
        if attribute['is_negation']:
            attribute_name = handle_negation(attribute_name, caches)

        result.append(check_attribute(image_pil, my_object['name'], detection, attribute_name, models, caches, negate=attribute['is_negation']))

    return result


#return dict mapping name to list of detections
def merge_detections(ids, objects, matched_detections):
    name2detections = {}
    for k in ids:
        name = objects[k]['name']
        if name not in name2detections:
            name2detections[name] = []

        if objects[k]['is_group']:
            name2detections[name].extend(matched_detections[k])
        elif matched_detections[k] is not None:
            name2detections[name].append(matched_detections[k])

    return name2detections


def check_count_comparison_relationship(relationship, objects, matched_detections, caches, logger):
    crocodile = handle_count_comparison(relationship['predicate'], caches, logger)
    if crocodile == 'N/A':
        return []

    assert(len(relationship['subject_ids']) == 1 and len(relationship['subject_ids']) == 1)
    subject_id, object_id = relationship['subject_ids'][0], relationship['object_ids'][0]
    assert(objects[subject_id]['is_group'] and objects[object_id]['is_group'])
    subject_count = len(matched_detections[subject_id])
    object_count = len(matched_detections[object_id])
    if crocodile == '>':
        answer = int(subject_count > object_count)
    else:
        assert(crocodile == '<')
        answer = int(subject_count < object_count)

    description = 'Check count comparison relationship: %ss[%d] %s %ss[%d]'%(objects[subject_id]['name'], subject_count, crocodile, objects[object_id]['name'], object_count)
    return [(description, answer)]


def check_relationship(image_pil, relationship, objects, matched_detections, models, caches, logger):
    if relationship['type'] == 'count comparison':
        return check_count_comparison_relationship(relationship, objects, matched_detections, caches, logger)

    result = []
    predicate = relationship['predicate']

    subject_name2detections = merge_detections(relationship['subject_ids'], objects, matched_detections)
    object_name2detections = merge_detections(relationship['object_ids'], objects, matched_detections)

    #I'm 99% sure this will always be true
    assert(len(subject_name2detections.keys()) == 1)
    assert(len(object_name2detections.keys()) == 1)

    for subject_name in sorted(subject_name2detections.keys()):
        for object_name in sorted(object_name2detections.keys()):
            if relationship['type'] in ['spatial only', 'negative spatial only']:
                if relationship['type'] == 'negative spatial only':
                    predicate = handle_negation(predicate, caches)

                spatial_vocab_entry, flip = handle_spatial_only_relationship(predicate, caches)
                subject_name_spatial, object_name_spatial = subject_name, object_name
                subject_name2detections_spatial, object_name2detections_spatial = subject_name2detections, object_name2detections
                if flip:
                    subject_name_spatial, object_name_spatial = object_name, subject_name
                    subject_name2detections_spatial, object_name2detections_spatial = object_name2detections, subject_name2detections

                result.append(check_predefined_spatial_relationship(image_pil, subject_name_spatial, subject_name2detections_spatial[subject_name_spatial], object_name_spatial, object_name2detections_spatial[object_name_spatial], spatial_vocab_entry, models, logger, negate=(relationship['type'] == 'negative spatial only')))
            elif relationship['type'] in ['spatial semantic', 'negative spatial semantic']:
                if relationship['type'] == 'negative spatial semantic':
                    predicate = handle_negation(predicate, caches)

                spatial_vocab_entry, flip = extract_spatial_from_spatial_semantic_relationship(predicate,subject_name,object_name,caches,logger)
                subject_name_spatial, object_name_spatial = subject_name, object_name
                subject_name2detections_spatial, object_name2detections_spatial = subject_name2detections, object_name2detections
                if flip:
                    subject_name_spatial, object_name_spatial = object_name, subject_name
                    subject_name2detections_spatial, object_name2detections_spatial = object_name2detections, subject_name2detections

                result.append(check_semantic_relationship(image_pil, subject_name, subject_name2detections[subject_name], object_name, object_name2detections[object_name], predicate, models, caches, logger, negate=(relationship['type'] == 'negative spatial semantic')))
                if spatial_vocab_entry != 'N/A':
                    result.append(check_predefined_spatial_relationship(image_pil, subject_name_spatial, subject_name2detections_spatial[subject_name_spatial], object_name_spatial, object_name2detections_spatial[object_name_spatial], spatial_vocab_entry, models, logger, negate=(relationship['type'] == 'negative spatial semantic')))
            elif relationship['type'] == 'attribute comparison':
                attribute, crocodile = extract_attribute_from_attribute_comparison_relationship(predicate, caches, logger)
                if attribute != 'N/A':
                    result.append(check_attribute_comparison_relationship(image_pil, subject_name, subject_name2detections[subject_name], object_name, object_name2detections[object_name], attribute, crocodile, models, caches))

            else:
                assert(False)

    return result


def get_image_filename(prompt, d_image, logger=None):
    matching_images = [d_image[k]['images']['DALLE_3']['filename'] for k in sorted(d_image.keys()) if d_image[k]['prompt'].replace('.','').strip().lower() == prompt.replace('.','').strip().lower()]
    assert(len(matching_images) == 1)
    image_filename = adjust_impath(matching_images[0])
    if logger is not None:
        logger.log(image_filename)

    return image_filename


def print_match(match, objects, image_pil, logger):
    for k in sorted(match.keys()):
        if match[k] is None:
            logger.log('%s{%s} ==> None'%(objects[k]['name'], str(k)))
        else:
            logger.log('%s{%s} ==> %s'%(objects[k]['name'], str(k), make_bbox_description(image_pil, objects[k]['name'], match[k])))


def is_valid_for_count_comparison(id_list, objects):
    return (len(id_list) == 1 and objects[id_list[0]]['is_group'] == 'Yes')


#return True if we should regenerate, False otherwise
def should_regenerate_semantic_graph(semantic_graph, logger):
    semantic_graph = json.loads(semantic_graph)
    if 'objects' not in semantic_graph or 'relationships' not in semantic_graph or 'scene' not in semantic_graph:
        logger.log('Missing "objects" or "relationships" or "scene" section')
        return True

    if 'description' not in semantic_graph['scene'] or 'absent_object_names' not in semantic_graph['scene'] or not isinstance(semantic_graph['scene']['absent_object_names'], list):
        logger.log('Bad scene:')
        logger.log(semantic_graph['scene'])
        return True

    objects, relationships = semantic_graph['objects'], semantic_graph['relationships']
    objects = {my_object['object_id'] : my_object for my_object in objects}
    for r in relationships:
        #NOTE: if there's really subject-object ambiguity then you can just put everything as both subject and object. We can skip self-loops now.
        if len(r['subject_ids']) == 0 or len(r['object_ids']) == 0:
            logger.log('Found bad relationship (empty subj or obj):')
            logger.log(r)
            return True

        if r['type'] == 'count comparison' and not (is_valid_for_count_comparison(r['subject_ids'], objects) and is_valid_for_count_comparison(r['object_ids'], objects)):
            logger.log('Found bad relationship (invalid count comparison):')
            logger.log(r)
            return True

    return False


def do_scene_checks(scene, image_pil, models, logger):
    results = []
    if scene['description'].lower().strip() != 'n/a':
        results.append(check_scene_description(image_pil, scene['description'], models, logger))
    else:
        logger.log('No scene/style description specified, no need to check.')

    if len(scene['absent_object_names']) > 0:
        logger.log('Objects %s specified as absent, use detector to check for them...'%(str(scene['absent_object_names'])))
        detections = run_object_detector(image_pil, scene['absent_object_names'], models)
        for name in scene['absent_object_names']:
            logger.log('-Detected %d instances of "%s" in image'%(len(detections[name]), name))
            results.append(('Check that there are no instances of "%s" detected in image'%(name), int(len(detections[name]) == 0)))
    else:
        logger.log('No absent objects specified, no need to check.')

    return results


def process_input(my_input, d_image, models, logger):
    logger.log(my_input)
    caches = {'semantic_relationship_cache':{}, 'attribute_cache':{}, 'attribute_comparison_relationship_cache':{}, 'LLM_cache':{}}
    image_filename = get_image_filename(my_input, d_image, logger=logger)
    image_pil = Image.open(image_filename).convert('RGB')
    candidate_graphs = []
    for graph_rep in range(NUM_FOR_MEDIAN_LENGTH):
        logger.log('Generating candidate graph %d...'%(graph_rep))
        while True:
            candidate_graph, (assistant_replyA, assistant_replyB, assistant_replyC) = generate_semantic_graph(my_input)
            if not should_regenerate_semantic_graph(candidate_graph, logger):
                logger.log('Semantic graph is good enough, no need to regenerate!')
                break

            logger.log('Regenerating semantic graph...')


        candidate_graphs.append(candidate_graph)

    logger.log('Candidate graph json colon-counts are: %s'%(str([g.count(':') for g in candidate_graphs])))
    semantic_graph = sorted(candidate_graphs, key=lambda g: g.count(':'))[NUM_FOR_MEDIAN_LENGTH//2]

    logger.log('')
    logger.log('SEMANTIC GRAPH:')
    logger.log(semantic_graph)
    logger.log('')
    logger.log('RUNNING OBJECT DETECTOR...')
    objects, relationships, scene = parse_semantic_graph(semantic_graph)
    object_names = sorted(set([objects[k]['name'] for k in sorted(objects.keys())]))
    detections = run_object_detector(image_pil, object_names, models)
    for name in object_names:
        logger.log('Detected %d instances of "%s" in image'%(len(detections[name]), name))

    count_result = do_count_checks(objects, detections)
    scene_result = do_scene_checks(scene, image_pil, models, logger)
    base_result = count_result + scene_result
    best_match = None
    best_result = None
    best_score = float('-inf')
    logger.log('EVALUATING MATCHES...')
    for match in match_generator(objects, detections):
        logger.log('CANDIDATE MATCH:')
        print_match(match, objects, image_pil, logger)
        logger.log('CHECKLIST RESULTS FOR CANDIDATE MATCH:')
        result = copy.deepcopy(base_result)
        for k in sorted(objects.keys()):
            if k not in objects or k not in match:
                logger.log(objects)
                logger.log(relationships)
                logger.log(detections)
                logger.log(match)
                assert(False)

            result.extend(check_object(image_pil, objects[k], match[k], models, caches))

        for relationship in relationships:
            result.extend(check_relationship(image_pil, relationship, objects, match, models, caches, logger))

        score = np.mean([p[1] for p in result])
        if score > best_score:
            best_score = score
            best_match = copy.deepcopy(match)
            best_result = copy.deepcopy(result)

        logger.log(result, pretty=True)
        logger.log('score = %s'%(str(score)))
        logger.log('')

    logger.log('BEST MATCH:')
    print_match(best_match, objects, image_pil, logger)
    logger.log('CHECKLIST RESULTS FOR BEST MATCH:')
    logger.log(best_result, pretty=True)
    logger.log('best_score = %s'%(str(best_score)))
    logger.log('')


def get_dst_image_filename(image_filename):
    basename = os.path.basename(image_filename)
    dirname = os.path.basename(os.path.dirname(image_filename))
    return os.path.join('example_genaibench_images', dirname, basename)


def get_log_filename(my_input, image_filename):
    prompt_part = re.split(r'[^A-Za-z0-9]+', my_input)
    prompt_part = [x for x in prompt_part if len(x) > 0]
    prompt_part = '_'.join(prompt_part)
    image_base = os.path.splitext(os.path.basename(image_filename))[0]
    image_dir = os.path.basename(os.path.dirname(image_filename))
    log_filename = os.path.join('example_genaibench_ontology_logs', image_dir + '_' + image_base + '_' + prompt_part + '.txt')
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    return log_filename


def reporting_stuff(my_input, d_image, logger):
    image_filename = get_image_filename(my_input, d_image)
    dst_image_filename = get_dst_image_filename(image_filename)
    if not os.path.exists(dst_image_filename):
        os.makedirs(os.path.dirname(dst_image_filename), exist_ok=True)
        shutil.copy(image_filename, dst_image_filename)

    log_filename = get_log_filename(my_input, image_filename)
    logger.save_and_clear(log_filename)


def main():
    logger = Logger()
    d_image = load_human_ratings_data()
    models = setup_models()
    for t, my_input in enumerate(MY_INPUTS):
        logger.log('EXAMPLE %d:'%(t))
        process_input(my_input, d_image, models, logger)
        logger.log('')
        logger.log('')
        reporting_stuff(my_input, d_image, logger)


if __name__ == '__main__':
    main()
