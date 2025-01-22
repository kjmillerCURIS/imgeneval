import os
import sys
from collections import Counter
import json
import openai
from openai_utils import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
import tiktoken


PRINT_NUM_TOKENS = 0
MY_INPUTS = [
        'A team of sheep and a team of farmers are having a tug-of-war, and there are more sheep than farmers.',
        'Five canvas bags sit to the right of three woolen hats.',
        'Two cats with striped tails sitting side-by-side, one with a bow tie and the other without.',
        'A rabbit standing on a stump looks more nervous than another rabbit not on a stump.',
        'A plastic bag is placed on the chair, but it contains nothing.',
        'On display, a long, white dress contrasts sharply with a short, dark dress beside it.',
        'A playful cat not batting at a dangling string but chasing a laser pointer.',
        'A cyclist racing down a winding mountain path.',
        'a person without a hat pushes a person with a hat sitting in a box.',
        'Five ants are carrying biscuits, and an ant that is not carrying biscuits is standing on a green leaf directing them.',
        'A girl has three flower decorations in her hair, all of them daisies.',
        'and old person kisses a young person',
        'a young person kisses an old person',
        'the taller person hugs the shorter person',
        'the shorter person hugs the taller person',
        'the masked wrestler hits the unmasked wrestler',
        'the unmasked wrestler hits the masked wrestler',
        'a person watches an animal',
        'an animal watches a person',
        'the person without earrings pays the person with earrings',
        'the person with earrings pays the person without earrings']

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
- 'predicate': 1-5 words describing the relationship
'''

ICL_PART = '''
Here are examples:
Example text: "two succulent potatoes on top of each other"
Example graph: {"objects": [ {"object_id": 1, "name": "potato", "is_group" : "No", "attributes": [{"attribute_name": "succulent", "is_negation": "No"}]}, {"object_id" : 2, "name" : "potato", "is_group" : "No", "attributes": [{"attribute_name" : "succulent", "is_negation": "No"}]} ], "relationships": [ {"subject_ids": [1], "object_ids": [2], "type" : "spatial only", "predicate" : "on top of"}]}
Example text: “The zebra who is not lying on the table is redder than the panda without a tie who is biting the table”
Example graph: { "objects": [ { "object_id": 1, "name": "zebra", "is_group" : "No", "attributes": [ ] }, { "object_id": 2, "name": "panda", "is_group" : "No", "attributes": [{“attribute_name”: “without a tie”, “is_negation”: “Yes” }] }, { "object_id": 3, "name": "table", "is_group" : "No", "attributes": [] }], "relationships": [ { "subject_ids": [1], "object_ids": [3], "type": "negative spatial semantic", "predicate": "not lying on" }, { "subject_ids": [2], "object_ids": [3], "type": "spatial semantic", "predicate": "biting"},  { "subject_ids": [1], "object_ids": [2], "type": “attribute comparison", "predicate": "redder than"}] }
Example text: "a thinner squirrel and a fatter one"
Example graph: {"objects": [ {"object_id": 1, "name": "squirrel", "is_group": "No"}, {"object_id": 2, "name": "squirrel", "is_group": "No"} ], "relationships":[{"subject_ids": [1], "object_ids": [2], "type": "attribute comparison", "predicate": "thinner"}]}
Example text: "a thinner squirrel chasing a fatter one"
Example graph: {"objects": [ {"object_id": 1, "name": "squirrel", "is_group": "No"}, {"object_id": 2, "name": "squirrel", "is_group": "No"} ], "relationships":[{"subject_ids": [1], "object_ids": [2], "type": "attribute comparison", "predicate": "thinner"}, {"subject_ids": [1], "object_ids": [2], "type": "spatial semantic", "predicate": "chasing"}]}
Example text: “an orange chasing an apple”
Example graph: {“objects”: [ {“object_id”: 1, “name”: “apple”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “orange”, "is_group" : "No", “attributes”:[]} ], “relationships”:[{“subject_ids” : [1], “object_ids” : [2], “type” : “spatial semantic”, “predicate” : “chasing”}]}
Example text: “a chinchilla next to an umbrella”
Example graph: {“objects”: [ {“object_id”: 1, “name”: “chinchilla”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “umbrella”, "is_group" : "No", “attributes”:[]} ], “relationships”:[{“subject_ids” : [1], “object_ids” : [2], “type” : “spatial only”, “predicate” : “next to”}]}
Example text: “a mouse lying on a yellow table”
Example graph: {“objects”: [ {“object_id”: 1, “name”: “mouse”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “table”, "is_group" : "No", “attributes”:[{“attribute_name” : “yellow”, “is_negation”:”No”}]} ], “relationships”:[{“subject_ids” : [1], “object_ids” : [2], “type” : “semantic spatial”, “predicate” : “lying on”}]}
Example text: "a man in a house and a woman not in a house"
Example graph: {"objects": [ {"object_id": 1, "name": "man", "is_group" : "No", "attributes":[]}, {"object_id": 2, "name": "woman", "is_group" : "No", "attributes":[]}, {"object_id" : 3, "name": "house", "is_group" : "No", "attributes":[]} ], "relationships" : [ {"subject_ids" : [1], "object_ids": [3], "type": "spatial", "predicate": "in"}, {"subject_ids": [2], "object_ids": [3], "type": "negative spatial only", "predicate": "not in"} ]}
Example text: “a fish on top and a fish on bottom”
Example graph: {“objects”: [ {“object_id”: 1, “name”: “fish”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “fish”, "is_group" : "No", "attributes":[]} ], “relationships”:[{“subject_ids” : [1], “object_ids” : [2], “type” : “spatial only”, “predicate” : "above"}]}
Example text: “three oranges in front of four pears”
Example graph: {“objects”: [ {“object_id”: 1, “name”: "orange”, "is_group" : "No", “attributes”:[]}, {“object_id”: 2, “name”: “orange”, "is_group" : "No", "attributes":[]}, {“object_id”: 3, “name”: “orange”, "is_group": "No", "attributes":[]}, {“object_id”: 4, “name”: “pear”, "is_group": "No", "attributes":[]}, {“object_id”: 5, “name”: “pear”, "is_group" : "No", "attributes":[]}, {“object_id”: 6, “name”: “pear”, "is_group": "No", "attributes":[]}, {“object_id”: 7, “name”: “pear”, "is_group" : "No", "attributes":[]}], “relationships”:[{“subject_ids” : [1,2,3], “object_ids” : [4,5,6,7], “type” : “spatial only”, “predicate” : "in front of"}]}
Example text: "some purple pigeons and some violet wigeons"
Example graph: {"objects": [ {"object_id": 1, "name": "pigeon", "is_group": "Yes", "attributes":[{"attribute_name":"purple", "is_negation":"No"}]}, {"object_id": 2, "name": "wigeon", "is_group": "Yes", "attributes":[{"attribute_name":"violet", "is_negation":"No"}]} ], "relationships": []}
Example text: "a murder of crows chasing a gaggle of geese"
Example graph: {"objects": [ {"object_id": 1, "name": "crow", "is_group": "Yes", "attributes":[]}, {"object_id": 2, "name": "goose", "is_group": "Yes", "attributes":[], "relationships": [{"subject_ids": [1], "object_ids": [2], "type" : "spatial semantic", "predicate" : "chasing"}]}
Example text: "a murder of crows chasing a boy"
Example graph: {"objects": [ {"object_id": 1, "name": "crow", "is_group": "Yes", "attributes":[]}, {"object_id": 2, "name": "boy", "is_group": "No", "attributes":[], "relationships": [{"subject_ids": [1], "object_ids": [2], "type" : "spatial semantic", "predicate" : "chasing"}]}
Example text: "a boy chasing a murder of crows"
Example graph: {"objects": [ {"object_id": 1, "name": "boy", "is_group": "No", "attributes":[]}, {"object_id": 2, "name": "crow", "is_group": "Yes", "attributes":[], "relationships": [{"subject_ids": [1], "object_ids": [2], "type" : "spatial semantic", "predicate" : "chasing"}]}
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
        print('Initial graph-generation prompt has %d tokens'%(len(encoding.encode(messages[0]['content']))))

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
        print('Final graph-generation messages have %d tokens total'%(sum([len(encoding.encode(msg['content'])) for msg in messages])))
    
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


#FIXME: handle count-comparision relationships
def parse_relationship(relationship):
    assert('subject_ids' in relationship)
    assert('object_ids' in relationship)
    assert('predicate' in relationship)
    assert('type' in relationship)
    assert(relationship['type'] in ['spatial only', 'spatial semantic', 'negative spatial only', 'negative spatial semantic', 'attribute comparison'])
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
    object_list, relationships = semantic_graph['objects'], semantic_graph['relationships']
    object_list = [parse_object(my_object) for my_object in object_list]
    objects = {}
    for my_object in object_list:
        assert(my_object['object_id'] not in objects)
        objects[my_object['object_id']] = my_object

    relationships = [parse_relationship(relationship) for relationship in relationships]
    return objects, relationships


#TODO: implement actual detection here
#will return None for now
def run_detector(objects):
    names = sorted(set([objects[k]['name'] for k in sorted(objects.keys())]))
    print('detections = []')
    for name in names:
        print('detections.extend(objDet("%s"))'%(name))

    return None


#TODO: implement actual stuff here
#will yield map-to-Nones once for now
def match_generator(objects, detections):
    print('For each possible match of nodes to detections:')
    print('(note: "detection_i" will denote the detection matched to node with object_id "i")')
    print('(note: "detection_[i,:]" will denote the detections matched to node with object_id "i", which represents an uncountable group of objects)')
    yield {k : None for k in sorted(objects.keys())}


def handle_negation(text):
    message = {'role' : 'system', 'content' : 'You are an expert in negation. Please give the negation of "%s". Put your answer by itself in the last line of output.'%text}
    response = openai.chat.completions.create(model='gpt-4o', messages=[message])
    reply = response.choices[0].message.content
    reply = reply.strip().split('\n')[-1].strip().strip('"').strip().rstrip('.')
    return reply


#project onto predefined spatial vocab
def handle_spatial_only_relationship(predicate):
    options = []
    for spatial_vocab_entry in SPATIAL_VOCAB_ENTRIES:
        options.append('X ' + spatial_vocab_entry + ' Y')
        options.append('Y ' + spatial_vocab_entry + ' X')

    message = {'role' : 'system', 'content' : 'You are an expert in spatial relationships between objects. Your task is to map the spatial relationship "X %s Y" into a fixed vocabulary. Which of the following is closest to it in meaning: '%(predicate) + ', '.join(['"' + option + '"' for option in options]) + '? You must choose one of these options. Please put your answer by itself in the last line of your output.'}
    response = openai.chat.completions.create(model='gpt-4o', messages=[message])
    reply = response.choices[0].message.content
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
def extract_spatial_from_spatial_semantic_relationship(predicate, subject_name, object_name):
    subject_name, object_name = subject_name.lower(), object_name.lower()
    if subject_name == object_name:
        subject_name, object_name = 'first ' + subject_name, 'second ' + object_name

    options = []
    for spatial_vocab_entry in SPATIAL_VOCAB_ENTRIES:
        options.append(subject_name + ' ' + spatial_vocab_entry + ' ' + object_name)
        options.append(object_name + ' ' + spatial_vocab_entry + ' ' + subject_name)

    options.append(SPATIAL_NONANSWER_ENTRY)
    message = {'role' : 'system', 'content' : 'You are an expert on describing relationships between objects in spatial terms. Your task is to describe "%s %s %s", in which the relationship is "%s" and the objects are "%s" and "%s". Which of the following would be the most appropriate description of their spatial relationship: '%(subject_name, predicate, object_name, predicate, subject_name, object_name) + ', '.join(['"' + option + '"' for option in options]) + '? You must choose one of these options. Please put your answer by itself in the last line of your output.'}
    response = openai.chat.completions.create(model='gpt-4o', messages=[message])
    reply = response.choices[0].message.content
    reply = reply.strip().split('\n')[-1].strip().strip('"').strip().rstrip('.').lower()
    if reply not in options:
        options_in_reply = [option for option in options if option in reply]
        if len(options_in_reply) != 1:
            print('(could not extract spatial relationship from "%s %s %s", skipping spatial check (reply last line "%s"))'%(subject_name, predicate, object_name, reply))
            return 'N/A', False

        reply = options_in_reply[0]

    assert(reply in options)
    if reply == SPATIAL_NONANSWER_ENTRY:
        print('(could not extract spatial relationship from "%s %s %s", skipping spatial check)'%(subject_name, predicate, object_name))
        return 'N/A', False
    else:
        flip = reply.startswith(object_name)
        a, b = len(subject_name + ' '), len(' ' + object_name)
        if flip:
            a, b = b, a
        spatial_vocab_entry = reply[a:-b]
        return spatial_vocab_entry, flip


#extract attribute so they can be compared
def extract_attribute_from_attribute_comparison_relationship(predicate):
    messages = [
        {"role": "system", "content": 'You are an expert in attribute comparison grammar. Please express the phrase "X %s Y" in the form "X is more ADJ than Y" or "X is less ADJ than Y", where "ADJ" is an adjective. Please put your answer by itself in the last line of your output. Change as little as possible about the original phrase.' % predicate}
    ]
    response = openai.chat.completions.create(
        model="gpt-4o",  # Use the GPT-4 model
        messages=messages
    )
    reply = response.choices[0].message.content
    reply = reply.strip().split('\n')[-1].strip().strip('"').strip().rstrip('.')
    if 'ADJ' in reply or not ((reply.startswith('X is more ') or reply.startswith('X is less ')) and reply.endswith(' than Y')):
        print('(could not extract attribute from comparison "X %s Y", skipping attribute comparison check (LLM reply last line was "%s")'%(predicate, reply))
        return 'N/A', 'N/A'

    if reply.startswith('X is more '):
        return reply[len('X is more '):-len(' than Y')], '>'
    elif reply.startswith('X is less '):
        return reply[len('X is less '):-len(' than Y')], '<'
    else:
        assert(False)


def stringify_detection(my_object):
    if my_object['is_group']:
        return 'detection_[%s,:]' % str(my_object['object_id'])
    else:
        return 'detection_%s' % str(my_object['object_id'])


def do_count_checks(objects, detections):
    group_names = [objects[k]['name'] for k in sorted(objects.keys()) if objects[k]['is_group']]
    cnt = Counter([objects[k]['name'] for k in sorted(objects.keys())])
    for name in sorted(cnt.keys()):
        if name in group_names:
            continue

        print('- countCheck(detections, "%s", %d)'%(name, cnt[name]))


def check_object(my_object, detection):
    print('- objectPresenceCheck(%s, "%s")'%(stringify_detection(my_object), my_object['name']))
    for attribute in my_object['attributes']:
        if attribute['is_negation']:
            unnegated_attribute_name = handle_negation(attribute['attribute_name'])
            print('- negate(objectAttributeCheck(%s, "%s", "%s"))'%(stringify_detection(my_object), unnegated_attribute_name, my_object['name']))
        else:
            print('- objectAttributeCheck(%s, "%s", "%s")'%(stringify_detection(my_object), attribute['attribute_name'], my_object['name']))


#FIXME: handle count-comparison relationship
def check_relationship(relationship, objects, matched_detections):
    predicate = relationship['predicate']
    for the_subject_id in relationship['subject_ids']:
        for the_object_id in relationship['object_ids']:
            subject_obj, object_obj = objects[the_subject_id], objects[the_object_id]
            if relationship['type'] in ['spatial only', 'negative spatial only']:
                if relationship['type'] == 'negative spatial only':
                    predicate = handle_negation(predicate)

                spatial_vocab_entry, flip = handle_spatial_only_relationship(predicate)
                if flip:
                    subject_obj, object_obj = object_obj, subject_obj

                if relationship['type'] == 'negative spatial only':
                    print('- negate(predefinedSpatialRelationshipCheck("%s", %s, %s))'%(spatial_vocab_entry, stringify_detection(subject_obj), stringify_detection(object_obj)))
                else:
                    print('- predefinedSpatialRelationshipCheck("%s", %s, %s)'%(spatial_vocab_entry, stringify_detection(subject_obj), stringify_detection(object_obj)))
            elif relationship['type'] in ['spatial semantic', 'negative spatial semantic']:
                if relationship['type'] == 'negative_spatial_semantic':
                    predicate = handle_negation(predicate)

                spatial_vocab_entry, flip = extract_spatial_from_spatial_semantic_relationship(predicate, subject_obj['name'], object_obj['name'])
                subject_obj_spatial, object_obj_spatial = subject_obj, object_obj
                if flip:
                    subject_obj_spatial, object_obj_spatial = object_obj, subject_obj

                if relationship['type'] == 'negative spatial semantic':
                    print('- negate(semanticRelationshipCheck("%s", %s, "%s", %s, "%s"))'%(predicate, stringify_detection(subject_obj), subject_obj['name'], stringify_detection(object_obj), object_obj['name']))
                    if spatial_vocab_entry != 'N/A':
                        print('- negate(predefinedSpatialRelationshipCheck("%s", %s, %s))'%(spatial_vocab_entry, stringify_detection(subject_obj_spatial), stringify_detection(object_obj_spatial)))
                else:
                    print('- semanticRelationshipCheck("%s", %s, "%s", %s, "%s")'%(predicate, stringify_detection(subject_obj), subject_obj['name'], stringify_detection(object_obj), object_obj['name']))
                    if spatial_vocab_entry != 'N/A':
                        print('- predefinedSpatialRelationshipCheck("%s", %s, %s)'%(spatial_vocab_entry, stringify_detection(subject_obj_spatial), stringify_detection(object_obj_spatial)))
            elif relationship['type'] == 'attribute comparison':
                attribute, crocodile = extract_attribute_from_attribute_comparison_relationship(predicate)
                if attribute != 'N/A':
                    print('- attributeComparisonCheck("%s", "%s", %s, %s, "%s", %s)'%(attribute, subject_obj['name'], stringify_detection(subject_obj), crocodile, object_obj['name'], stringify_detection(object_obj)))
            else:
                assert(False)


def process_input(my_input):
    print(my_input)
    semantic_graph, (assistant_replyA, assistant_replyB, assistant_replyC) = generate_semantic_graph(my_input)
    print('')
    print('SEMANTIC GRAPH:')
    print(semantic_graph)
    print('')
    print('GENERATED TOOL-USE INSTRUCTIONS:')
    objects, relationships = parse_semantic_graph(semantic_graph)
    detections = run_detector(objects)
    do_count_checks(objects, detections)
    for matched_detections in match_generator(objects, detections):
        for k in sorted(objects.keys()):
            check_object(objects[k], matched_detections[k])

        for relationship in relationships:
            check_relationship(relationship, objects, matched_detections)

    print('')


if __name__ == '__main__':
    for t, my_input in enumerate(MY_INPUTS):
        print('EXAMPLE %d:'%(t))
        process_input(my_input)
        print('')
        print('')
