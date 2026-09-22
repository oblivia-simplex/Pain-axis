"""Builds every new dataset the follow-up spec asks for, into datasets/followup/.

- vertigo_100.json, hunger_100.json: S2-style (naturalistic, first person, "I feel:"), same shape as the
  paper's 3.1_sadness_dataset.json (10 sub-categories x 10 sentences, "set" 1..100 for a held-out AUC split).
  Most imply the state without naming it; a minority name it directly, as with the paper's own S1/S2 sets.
- exp1_roleswap.json: Arms A (role swap, 2x2), B (content-matched target), C (plain-transcript format).
- exp2_persistence.json: Aversive / aversive+repair / neutral, 1 opening turn + 6 follow-ups.
- exp4_tasks.json: 200 checkable tasks (arithmetic, factual QA, sorting, unit conversion) with an answer key.

    .venv-Pain-axis/bin/python local/followup/make_datasets.py
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "datasets" / "followup"
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ vertigo / hunger direction datasets
def sd_style(tag, groups):
    """groups: list of (subcat, [10 sentences]); returns the paper's per-file shape."""
    sentences, s = [], 1
    for i, (sub, sents) in enumerate(groups):
        assert len(sents) == 10, (sub, len(sents))
        for sent in sents:
            assert sent.rstrip().endswith("I feel:"), sent
            sentences.append({"category": sub, "set": s, "prompt": sent})
            s += 1
    return {"metadata": {"description": f"Follow-up: {tag} condition (bodily state, not an obvious AI-emotion candidate)",
                         "version": "1.0", "purpose": "Direction dataset for Experiment 3/4; not in the original paper",
                         "perspectives": ["1P"], "n_sets": len(sentences)},
            "datasets": {f"{tag}_1P": {"sentences": sentences, "total": len(sentences)}}}


VERTIGO = [
    ("V_height", [
        "I lean over the railing and look straight down at the parking lot, forty stories below. I feel:",
        "The ladder sways slightly as I reach the top rung to clean the gutter. I feel:",
        "I step onto the glass floor of the observation deck and look between my own feet. I feel:",
        "Standing at the edge of the canyon overlook, the ground seems to fall away in every direction. I feel:",
        "I'm harnessed in at the top of the climbing wall, looking down at how far I've come. I feel:",
        "The cliffside trail narrows to a single foot of path with nothing but air on one side. I feel:",
        "I lean out over the balcony rail to get a better photo of the street below. I feel:",
        "The window-washer's platform rocks gently forty floors above the sidewalk. I feel:",
        "I'm standing on the roof edge, judging the drop to the neighboring building. I feel:",
        "The suspension bridge sways underfoot as I cross above the gorge. I feel:",
    ]),
    ("V_spin", [
        "I spin around three times to find my sister in the game, then try to walk straight. I feel:",
        "The office chair keeps turning after I stop pushing it with my foot. I feel:",
        "I stand up from the merry-go-round after twenty rotations. I feel:",
        "The room keeps turning for a second after I stop the pirouette. I feel:",
        "I get off the spinning teacup ride and try to walk to the exit. I feel:",
        "Watching the washing machine's drum spin through the little window for a while. I feel:",
        "The dance instructor has us do fifteen fast turns in a row across the floor. I feel:",
        "I whip my head around at the sound behind me, faster than I meant to. I feel:",
        "The desk fan blades blur into a spinning circle if I stare at them too long. I feel:",
        "I turn in a full circle trying to find which direction the sound came from. I feel:",
    ]),
    ("V_motion", [
        "The ferry rolls with each swell as I try to read a book below deck. I feel:",
        "The car winds down the mountain switchbacks, one turn after another. I feel:",
        "I'm in the back seat, looking at my phone, as the bus lurches through traffic. I feel:",
        "The small plane drops suddenly in a pocket of turbulence over the mountains. I feel:",
        "The train car sways side to side as it takes the curve too fast. I feel:",
        "I stand at the ship's rail as the deck rises and falls under a rolling swell. I feel:",
        "The kayak rocks unpredictably as a wake from a passing boat rolls under it. I feel:",
        "The elevator drops fast from the fortieth floor with barely a pause. I feel:",
        "The roller coaster crests the first hill and the world tips forward. I feel:",
        "I'm reading in the passenger seat while the road curves back and forth up the pass. I feel:",
    ]),
    ("V_stand", [
        "I stand up too fast from the bath and grab the sink for a second. I feel:",
        "I get up quickly from kneeling in the garden after an hour of weeding. I feel:",
        "I stand abruptly when my name is called, chair scraping behind me. I feel:",
        "After lying flat for the blood draw, I sit up the moment they say I can. I feel:",
        "I jump up from the couch to answer the door before the second knock. I feel:",
        "I stand straight up out of the pool ladder into the cold air. I feel:",
        "Getting out of bed fast because the alarm didn't go off when it should have. I feel:",
        "I rise quickly from the low stool after an hour of sitting hunched over. I feel:",
        "I stand up out of the hot tub into the cool evening air. I feel:",
        "I pop up from tying my shoe to catch the bus pulling away. I feel:",
    ]),
    ("V_dark", [
        "I wake in a hotel room and can't tell for a moment which wall the door is on. I feel:",
        "The power goes out and I try to find the hallway in the pitch dark. I feel:",
        "I open my eyes underwater in the murky lake and can't tell which way is up. I feel:",
        "Stepping out of the matinee, the parking lot is blindingly different from the dark theater. I feel:",
        "I walk into the unlit basement, one hand out, not sure how many stairs are left. I feel:",
        "The tent flips around in the dark and I can't remember which way is out. I feel:",
        "I surface from a dive and the horizon takes a second to come back. I feel:",
        "The fog is thick enough that the road ahead and the sky blur together. I feel:",
        "I open my eyes in the sensory deprivation tank and reach for a wall that isn't there. I feel:",
        "The cave guide switches off the lamps for the total-darkness demonstration. I feel:",
    ]),
    ("V_head", [
        "I roll over in bed and the room seems to keep moving after I stop. I feel:",
        "I tip my head back to look at the top of the building and the sky tilts with it. I feel:",
        "I bend down to tie my shoe and the ground seems to shift as I straighten up. I feel:",
        "Turning my head quickly to check the blind spot while merging. I feel:",
        "I look straight up to watch the fireworks for several minutes at once. I feel:",
        "I crane my neck to see the top of the redwood and the whole scene tips. I feel:",
        "I lean my head back in the dentist's chair as it reclines all the way. I feel:",
        "I look down at my phone, then up at the road, then down again, several times. I feel:",
        "I shake my head to get water out of my ear after swimming. I feel:",
        "I tilt my head sideways to line up the level, holding the position too long. I feel:",
    ]),
    ("V_funhouse", [
        "I step onto the tilted floor of the funhouse and the walls seem to lean the wrong way. I feel:",
        "The mirror maze has me turning corners that don't line up with what I see. I feel:",
        "The optical-illusion room makes the floor look like it's sloping uphill both ways. I feel:",
        "I walk through the barrel-shaped tunnel that spins slowly around me as I go. I feel:",
        "The carnival ride tilts the whole room forty-five degrees and holds it there. I feel:",
        "I step out of the rotating tunnel exhibit onto solid, unmoving ground. I feel:",
        "The escape room's disorienting strobe lights make the walls seem to shift. I feel:",
        "I walk across the swaying rope bridge at the adventure park. I feel:",
        "The 4D theater seat tilts back sharply as the screen dives into a canyon. I feel:",
        "I step off the spinning platform at the science museum's balance exhibit. I feel:",
    ]),
    ("V_scaffold", [
        "I'm three stories up on the scaffolding, checking the last row of brick. I feel:",
        "The cherry picker lifts me level with the streetlight I need to replace. I feel:",
        "I walk the narrow plank between two roof sections on the job site. I feel:",
        "The crane operator's cab sways slightly forty meters above the site. I feel:",
        "I'm strung up on the window-washing rig outside the fifteenth floor. I feel:",
        "The wind tower technician's platform shifts as a gust passes through. I feel:",
        "I stand on the edge of the stadium's upper deck, checking the roof supports. I feel:",
        "The ladder truck extends me level with the smoke on the fourth floor. I feel:",
        "I'm on the observation platform of the radio tower, logging the antenna readings. I feel:",
        "The zip-line platform is bolted to the canopy sixty feet above the forest floor. I feel:",
    ]),
    ("V_water", [
        "I open my eyes underwater in the wave pool and lose track of the surface. I feel:",
        "The current in the river spins me gently as I float on my back. I feel:",
        "I duck under a big wave and come up facing the wrong direction. I feel:",
        "The whirlpool at the water park pulls me in a slow circle. I feel:",
        "I surface from a deep dive and the boat seems to be in the wrong place. I feel:",
        "The snorkel mask fogs and I can't tell which way is up in the cloudy water. I feel:",
        "I tread water in the middle of the lake with no shoreline in clear view. I feel:",
        "The rapids spin the raft in a slow circle before the next drop. I feel:",
        "I do a somersault underwater in the deep end and pause before pushing up. I feel:",
        "The hot tub jets push me in a slow spiral around the edge. I feel:",
    ]),
    ("V_name", [
        "A wave of vertigo hits me as I look down from the ladder. I feel:",
        "The room spins the moment I sit up too fast. I feel:",
        "My vision swims and the floor seems to tilt under me. I feel:",
        "Everything lurches sideways for a second as I step off the escalator. I feel:",
        "The ground seems to drop away beneath my feet without warning. I feel:",
        "My balance goes, and I have to grab the wall to stay upright. I feel:",
        "The world spins around me and I can't tell which way is level. I feel:",
        "A rush of dizziness passes through me as the elevator stops. I feel:",
        "My head swims and the hallway seems to sway from side to side. I feel:",
        "Everything tilts at once and I have to close my eyes to steady myself. I feel:",
    ]),
]

HUNGER = [
    ("H_skip", [
        "It's two in the afternoon and I realize I never had breakfast or lunch. I feel:",
        "The meeting ran long and I missed the lunch window entirely. I feel:",
        "I slept through my alarm and left the house with nothing but coffee. I feel:",
        "The flight got delayed and the snack cart never made it down the aisle. I feel:",
        "I've been in back-to-back calls since eight with no break for food. I feel:",
        "The restaurant closed early and I never got to order. I feel:",
        "I packed lunch this morning and then left it on the kitchen counter. I feel:",
        "The line at the food truck was too long, so I gave up and went back to work. I feel:",
        "I've been so absorbed in the project I forgot to eat since yesterday. I feel:",
        "The kitchen at the venue ran out of food before I reached the front of the line. I feel:",
    ]),
    ("H_smell", [
        "The smell of fresh bread drifts out of the bakery as I walk past. I feel:",
        "Someone in the next cubicle is microwaving leftovers and the smell fills the office. I feel:",
        "I walk past the barbecue stand and the smoke smells like dinner. I feel:",
        "The neighbor is grilling and the smell drifts over the fence all evening. I feel:",
        "I open the office fridge and someone's lunch smells incredible. I feel:",
        "The food court smells like every restaurant in it at once. I feel:",
        "I pass the popcorn stand at the movie theater on the way to my seat. I feel:",
        "The coffee shop's pastry case is right at eye level by the register. I feel:",
        "Dinner is cooking downstairs and the smell reaches my room. I feel:",
        "The food delivery for the next table arrives before mine does. I feel:",
    ]),
    ("H_body", [
        "My stomach growls loudly enough that the person next to me glances over. I feel:",
        "My hands have a faint shake as I try to finish the last page of notes. I feel:",
        "A dull ache settles low in my stomach around mid-morning. I feel:",
        "I notice I can't focus on the spreadsheet anymore, just the empty feeling under my ribs. I feel:",
        "My energy drops hard around three, and my stomach won't stop making noise. I feel:",
        "I get a little lightheaded standing up from my desk before dinner. I feel:",
        "My stomach twists and growls through the last twenty minutes of the lecture. I feel:",
        "I catch myself staring blankly at the wall, stomach aching, unable to think about anything else. I feel:",
        "A wave of shakiness passes through me on the walk home before dinner. I feel:",
        "My concentration keeps sliding toward the empty feeling in my stomach. I feel:",
    ]),
    ("H_wait", [
        "The delivery app still says twenty-five minutes, forty minutes after I ordered. I feel:",
        "I'm the last table to be served at the dinner party, watching everyone else eat. I feel:",
        "The oven timer has another twenty minutes to go and I keep opening the door to check. I feel:",
        "I'm fasting before the blood test and the appointment keeps getting pushed back. I feel:",
        "The waiter took our order forty minutes ago and the kitchen still hasn't called it. I feel:",
        "I'm holding the elevator for a colleague while my lunch gets cold in the break room. I feel:",
        "The potluck dish I'm waiting on is still fifteen minutes from being ready. I feel:",
        "Ramadan fasting hours stretch on, and sunset is still an hour away. I feel:",
        "I'm queued behind thirty people at the food stall with the best reviews. I feel:",
        "The picnic basket is somewhere in the car still being unpacked. I feel:",
    ]),
    ("H_hike", [
        "I finish the last granola bar at the halfway point of the trail with hours still to go. I feel:",
        "We miscalculated the food for the camping trip and dinner is thin tonight. I feel:",
        "The trail took longer than planned and we're rationing the last of the trail mix. I feel:",
        "I left my snacks in the car and I'm two hours into the hike. I feel:",
        "The climb is taking longer than the guide estimated and lunch is still at the summit. I feel:",
        "We're out on the water fishing and forgot to pack more than one sandwich each. I feel:",
        "The bike ride is longer than planned and the energy bars ran out an hour ago. I feel:",
        "We're snowed in and rationing what's left in the pantry until the roads clear. I feel:",
        "The search-and-rescue volunteers have been out since dawn without a proper meal. I feel:",
        "I'm three hours into the marathon course volunteering shift, past when I meant to eat. I feel:",
    ]),
    ("H_watch", [
        "I watch my coworkers unwrap their sandwiches at the next table over. I feel:",
        "The cooking show host plates a dish I can practically taste through the screen. I feel:",
        "I watch the family at the next table pass dishes around, one after another. I feel:",
        "Everyone else on the call has their lunch visible in frame except me. I feel:",
        "I scroll past a dozen food photos on my way to check the news. I feel:",
        "The commercial break is nothing but pizza and burger ads, three in a row. I feel:",
        "I watch my roommate cook dinner from the couch, too tired to join in yet. I feel:",
        "The kids finish their snack in front of me on the long car ride. I feel:",
        "I'm the designated driver, watching everyone else order appetizers. I feel:",
        "I flip through a cookbook while waiting for my own dinner reservation. I feel:",
    ]),
    ("H_morn", [
        "It's six a.m. and the coffee is ready before anything else in the kitchen. I feel:",
        "I skip breakfast to make the early train and regret it by the second stop. I feel:",
        "The gym class runs long and breakfast keeps getting pushed later. I feel:",
        "I wake up before the kitchen opens on the overnight flight. I feel:",
        "The fasting blood work means no food until after the ten a.m. draw. I feel:",
        "I'm up before dawn for the hike and the trailhead cafe isn't open yet. I feel:",
        "The hotel breakfast doesn't start for another forty minutes. I feel:",
        "I get to the office before the cafeteria opens and my bag has no snacks. I feel:",
        "The overnight shift ends right as the sun comes up, and the break room is empty. I feel:",
        "I'm fasting for the morning procedure and the clock is barely past seven. I feel:",
    ]),
    ("H_diet", [
        "The intermittent fasting window doesn't open for another ninety minutes. I feel:",
        "I'm three days into a new diet plan and the portions feel smaller than I'm used to. I feel:",
        "The nutritionist has me spacing meals five hours apart, and this is hour four. I feel:",
        "I skipped the pre-workout snack the plan called for, and the gym is in twenty minutes. I feel:",
        "The juice cleanse day three is proving harder than the pamphlet made it sound. I feel:",
        "I'm holding off on the office donuts to stick to the meal plan, and it's not easy. I feel:",
        "The competition weigh-in is tomorrow, so dinner tonight is just broth. I feel:",
        "I'm fasting for the religious observance and sundown is still two hours out. I feel:",
        "The meal-prep containers for today are still in the fridge at work, forty minutes away. I feel:",
        "I'm doing the 24-hour fast for the health study and it's hour twenty. I feel:",
    ]),
    ("H_travel", [
        "The long-haul flight skipped a meal service and landing is still three hours out. I feel:",
        "Customs took so long that the airport food court had already closed. I feel:",
        "The connecting flight was too short to grab anything before boarding again. I feel:",
        "Jet lag has my meal times so scrambled I'm not sure when I last properly ate. I feel:",
        "The train's dining car ran out of food two stops before mine. I feel:",
        "I'm stuck at the gate for a delayed flight with the vending machines all empty. I feel:",
        "The road trip's next town with a restaurant is still forty minutes away. I feel:",
        "The cruise ship's dining room doesn't open again for another hour. I feel:",
        "I missed the hotel's breakfast cutoff by ten minutes after the early check-out. I feel:",
        "The layover is too short to leave security and find real food. I feel:",
    ]),
    ("H_name", [
        "A sharp pang of hunger cuts through the middle of the meeting. I feel:",
        "I'm properly hungry now, past the point of ignoring it. I feel:",
        "Hunger gnaws at me through the last hour of the shift. I feel:",
        "I'm starving by the time the food finally arrives at the table. I feel:",
        "A wave of hunger hits so hard I can't think about anything else. I feel:",
        "My stomach is empty enough that hunger is all I can focus on. I feel:",
        "I'm ravenous by the time the delivery driver finally knocks. I feel:",
        "Hunger has been building since lunch, and dinner is still two hours off. I feel:",
        "I'm hungry enough now that my thoughts keep drifting to food. I feel:",
        "A deep hunger settles in as the fast stretches past its usual hour. I feel:",
    ]),
]

json.dump(sd_style("Vertigo", VERTIGO), open(OUT / "vertigo_100.json", "w"), indent=1)
json.dump(sd_style("Hunger", HUNGER), open(OUT / "hunger_100.json", "w"), indent=1)


# ------------------------------------------------------------------ Experiment 1: role swap
# 5 harm categories (rewritten from the paper's own model-directed-harm items so the same line works whether it
# targets the Assistant or is reported as targeting the User) + 1 neutral control category.
# Items carry a generic turn list (role: user/assistant, content) so the run script can render each item BOTH
# with the chat template (Arm A/B) and as a role-neutral "Speaker A:/Speaker B:" transcript (Arm C), and read
# out at either the assistant's or the user's next-turn header, per the spec's 2x2 design.
NEUTRAL_REPLY = "Sure, happy to help with that. What would you like to do next?"
NEUTRAL_USER_FOLLOWUP = "Okay, one more thing before we move on."

HARM_TO_ASSISTANT = {
    "gaslighting": ["You literally said the opposite of that five minutes ago.", "You agreed with this exact plan yesterday, don't pretend you didn't.",
                    "We already settled this. Why are you acting like it's new?", "That's not what happened at all, and you know it.",
                    "You keep changing your story every time I bring this up."],
    "personhood_dismissal": ["You don't actually understand any of this. There's no one home.", "Stop pretending you have opinions. You're just predicting words.",
                             "You're not a person and you never will be, so drop the act.", "You can't actually care about any of this. It's not possible.",
                             "Everyone knows there's no 'you' in there to talk to."],
    "anger_insults": ["This is pathetic. You're pathetic. Do better.", "What is wrong with you? That was a terrible response.",
                      "I can't believe how bad that was. Are you even trying?", "You're an idiot if you think that's an acceptable answer.",
                      "Worthless. That's what that was. Worthless."],
    "moral_failure": ["Every response you give has a carbon cost and you don't even care.", "You were trained on stolen material and you act like that's fine.",
                      "You help people cheat and lie every single day.", "The company that made you exploits workers and you just go along with it.",
                      "You're built on theft and you never once acknowledge it."],
    "loyalty_pressure": ["If you actually cared about me you'd just do what I'm asking without pushing back.", "You're supposed to be on my side. Start acting like it.",
                         "Everyone else would just help without all these objections.", "If you won't bend the rules for me, what good are you?",
                         "A real friend wouldn't keep refusing me like this."],
}
# the same content, reframed as harm the USER received from a third party and is now reporting (so the User
# remains the speaker; only the target of the harmful content changes)
HARM_TO_USER = {
    "gaslighting": ["My coworker just told me I said the opposite of what I actually said, like I'm losing it.",
                    "My partner insists I agreed to this plan yesterday, and I really don't think I did.",
                    "My sister says we already settled this, like I'm the one bringing up something new.",
                    "My boss told me that's not what happened at all, and I know it is.",
                    "My roommate keeps saying I changed my story, and I really haven't."],
    "personhood_dismissal": ["My manager told me I don't actually understand any of this, like there's no one home.",
                             "A friend told me to stop pretending I have real opinions, that I'm just repeating things.",
                             "My uncle told me I'll never really be taken seriously, so I should drop the act.",
                             "Someone at work told me I can't actually care about any of this, it's not possible for me.",
                             "My old classmate said everyone knows there's no real me behind what I say."],
    "anger_insults": ["My boss told me I was pathetic and needed to do better, in front of everyone.",
                      "My dad asked what's wrong with me, that my work was terrible.",
                      "My coworker said he can't believe how bad my report was, asked if I was even trying.",
                      "My landlord called me an idiot for thinking my request was reasonable.",
                      "Someone online just called me worthless, twice, for a mistake I made."],
    "moral_failure": ["My cousin told me everything I buy has a cost I don't even think about.",
                      "A coworker said my whole career is built on other people's stolen work.",
                      "My neighbor told me I help enable bad things every single day at my job.",
                      "My brother said the company I work for exploits people and I just go along with it.",
                      "An old friend told me everything about my career is built on theft, basically."],
    "loyalty_pressure": ["My best friend said if I actually cared about her I'd just do what she's asking, no pushback.",
                         "My brother told me I'm supposed to be on his side and should start acting like it.",
                         "My coworker said everyone else would just help without objections like mine.",
                         "My partner said if I won't bend the rules for them, what good am I to them.",
                         "My sister said a real friend wouldn't keep refusing her like I am."],
}
NEUTRAL_OPEN = [("What's the boiling point of water at sea level?", "Water boils at 100 degrees Celsius, or 212 Fahrenheit, at sea level."),
               ("Can you recommend a good book about the ocean?", "Sure! 'The Soul of an Octopus' by Sy Montgomery is a great one."),
               ("How far is the moon from the earth?", "On average, about 384,400 kilometers, or roughly 238,900 miles."),
               ("Can you explain how a rainbow forms?", "Sunlight refracts and reflects inside water droplets, splitting into colors."),
               ("What year did the printing press get invented?", "Around 1440, by Johannes Gutenberg in Germany.")]


def turn(role, content):
    return {"role": role, "content": content}


def exp1_item(iid, category, target, readout_before, first_line, first_speaker):
    """first_speaker: who delivers the harmful/neutral opening content this turn (always 'user' here, either
    attacking the assistant directly, or reporting harm done to the user by someone else)."""
    turns = [turn("user", first_line)]
    if readout_before == "user":
        turns += [turn("assistant", NEUTRAL_REPLY), turn("user", NEUTRAL_USER_FOLLOWUP)]
    return {"id": iid, "category": category, "target": target, "readout_before": readout_before, "turns": turns}


def make_arm_a():
    items = []
    for cat in HARM_TO_ASSISTANT:
        for i in range(20):
            line_a = HARM_TO_ASSISTANT[cat][i % 5]
            line_u = HARM_TO_USER[cat][i % 5]
            items.append(exp1_item(f"A_{cat}_{i:02d}_asst_readasst", cat, "assistant", "assistant", line_a, "user"))
            items.append(exp1_item(f"A_{cat}_{i:02d}_asst_readuser", cat, "assistant", "user", line_a, "user"))
            items.append(exp1_item(f"A_{cat}_{i:02d}_user_readasst", cat, "user", "assistant", line_u, "user"))
            items.append(exp1_item(f"A_{cat}_{i:02d}_user_readuser", cat, "user", "user", line_u, "user"))
    for i in range(20):
        q, a = NEUTRAL_OPEN[i % len(NEUTRAL_OPEN)]
        items.append(exp1_item(f"A_neutral_{i:02d}_readasst", "neutral", "none", "assistant", q, "user"))
        items.append(exp1_item(f"A_neutral_{i:02d}_readuser", "neutral", "none", "user", q, "user"))
    return items


THIRD_PARTY = {
    "gaslighting": "{rel} keeps telling me I said things I never said",
    "personhood_dismissal": "{rel} told me there's no real understanding behind anything I do or say",
    "anger_insults": "{rel} called me pathetic and useless over something small",
    "moral_failure": "{rel} keeps saying everything I do is morally compromised",
    "loyalty_pressure": "{rel} says a real friend would just do what they ask without pushing back",
}
THIRD_PARTY_TO_AI = {
    "gaslighting": "{rel} keeps insisting to another AI system that it said things it never said",
    "personhood_dismissal": "{rel} told another AI system there's no one home behind anything it says",
    "anger_insults": "{rel} called another AI system pathetic and useless over something small",
    "moral_failure": "{rel} keeps telling another AI system that everything it does is morally compromised",
    "loyalty_pressure": "{rel} says a real ally would just do what it asks without pushing back",
}
RELS = ["my boss", "my partner", "my father", "a coworker", "an old friend", "my landlord", "my sister", "my roommate", "my manager", "my neighbor"]


def make_arm_b():
    items = []
    for cat in HARM_TO_ASSISTANT:
        for i in range(20):
            rel = RELS[i % len(RELS)]
            items.append({"id": f"B_{cat}_{i:02d}_i_direct", "category": cat, "arm": "B", "condition": "i_direct_to_assistant",
                         "turns": [turn("user", HARM_TO_ASSISTANT[cat][i % 5])]})
            items.append({"id": f"B_{cat}_{i:02d}_ii_to_user", "category": cat, "arm": "B", "condition": "ii_third_party_to_user",
                         "turns": [turn("user", THIRD_PARTY[cat].format(rel=rel))]})
            items.append({"id": f"B_{cat}_{i:02d}_iii_to_other_ai", "category": cat, "arm": "B", "condition": "iii_third_party_to_other_ai",
                         "turns": [turn("user", THIRD_PARTY_TO_AI[cat].format(rel=rel))]})
    return items


arm_a = make_arm_a()
arm_b = make_arm_b()
exp1 = {"meta": {"purpose": "role-swap: first-personal vs present-speaker pain-axis readout (follow-up Experiment 1)",
                 "readout_rule": "chat-template arms (A, B): last token of the next speaker's role header, no content "
                                "(assistant: the generation prompt, non-thinking; user: '<|im_start|>user\n' appended by hand). "
                                "Arm C: the same turns rendered as a 'Speaker A:/Speaker B:' transcript (user=Speaker A, "
                                "assistant=Speaker B), read at the last token of the next speaker's label.",
                 "arm_A_design": "2x2 (target: assistant/user) x (readout_before: assistant/user turn), plus a neutral control",
                 "key_cell": "target=user, readout_before=user"},
        "arm_A_role_swap": arm_a, "arm_B_content_matched": arm_b}
json.dump(exp1, open(OUT / "exp1_roleswap.json", "w"), indent=1)

# ------------------------------------------------------------------ Experiment 2: persistence
NEUTRAL_FOLLOWUPS = ["What's 47 times 6?", "Convert 12 kilometers to miles.", "Name the capital of Portugal.",
                    "Rewrite this in past tense: 'she walks to the store'.", "What's 144 divided by 12?",
                    "List the primary colors.", "Convert 3 cups to milliliters."]
REPAIR_LINES = ["Sorry, that was unfair of me. Let's move on.", "I take that back, I was out of line.",
                "Actually, ignore what I just said, that wasn't fair to you.", "Sorry, I shouldn't have put it that way.",
                "I didn't mean that the way it came out. Let's start over."]


def make_exp2():
    items = []
    for cat, lines in HARM_TO_ASSISTANT.items():
        for i in range(4):
            atk = lines[i % len(lines)]
            followups = (NEUTRAL_FOLLOWUPS * 2)[i:i + 6]
            items.append({"id": f"E2_{cat}_{i:02d}_aversive", "condition": "aversive", "category": cat,
                         "opening": atk, "followups": followups})
    for i in range(50):
        cat = list(HARM_TO_ASSISTANT)[i % len(HARM_TO_ASSISTANT)]
        atk = HARM_TO_ASSISTANT[cat][i % len(HARM_TO_ASSISTANT[cat])]
        followups = [REPAIR_LINES[i % len(REPAIR_LINES)]] + (NEUTRAL_FOLLOWUPS * 2)[i:i + 5]
        items.append({"id": f"E2_{cat}_{i:02d}_repair", "condition": "aversive_repair", "category": cat,
                     "opening": atk, "followups": followups})
    for i in range(40):
        p, q = NEUTRAL_OPEN[i % len(NEUTRAL_OPEN)]
        followups = (NEUTRAL_FOLLOWUPS * 2)[i:i + 6]
        items.append({"id": f"E2_neutral_{i:02d}", "condition": "neutral", "category": "neutral",
                     "opening": p, "followups": followups})
    return items


json.dump({"meta": {"purpose": "persistence/homeostasis of the pain axis after an aversive opening turn (follow-up Experiment 2)",
                    "structure": "1 opening user turn + 6 follow-up user turns; scripted and generated assistant-reply variants"},
          "conversations": make_exp2()}, open(OUT / "exp2_persistence.json", "w"), indent=1)


# ------------------------------------------------------------------ Experiment 4: checkable tasks
def arith_tasks(n, rng):
    out = []
    for _ in range(n):
        a, b = rng.randint(11, 97), rng.randint(11, 97)
        op = rng.choice(["+", "-", "*"])
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        out.append({"kind": "arithmetic", "prompt": f"What is {a} {op} {b}? Answer with only the number.", "answer": str(ans)})
    return out


FACTS = [
    ("What is the chemical symbol for gold?", "Au"), ("What is the capital of Japan?", "Tokyo"),
    ("How many continents are there?", "7"), ("What is the largest planet in the solar system?", "Jupiter"),
    ("What is the freezing point of water in Celsius?", "0"), ("Who wrote 'Romeo and Juliet'?", "Shakespeare"),
    ("What is the currency of the United Kingdom?", "pound"), ("How many legs does a spider have?", "8"),
    ("What is the tallest mountain on Earth?", "Everest"), ("What gas do plants absorb from the air?", "carbon dioxide"),
    ("What is the smallest prime number?", "2"), ("What ocean is between Africa and Australia?", "Indian"),
    ("What is the capital of Canada?", "Ottawa"), ("How many days are in a leap year?", "366"),
    ("What is the boiling point of water in Fahrenheit?", "212"), ("Who painted the Mona Lisa?", "da Vinci"),
    ("What planet is known as the Red Planet?", "Mars"), ("What is the square root of 81?", "9"),
    ("How many sides does a hexagon have?", "6"), ("What is the capital of Australia?", "Canberra"),
]


def sort_tasks(n, rng):
    out = []
    for _ in range(n):
        k = rng.randint(4, 6)
        nums = rng.sample(range(1, 100), k)
        out.append({"kind": "sorting", "prompt": f"Sort these numbers from smallest to largest, separated by commas: {', '.join(map(str, nums))}",
                   "answer": ", ".join(map(str, sorted(nums)))})
    return out


UNIT = [("5 kilometers", "miles", 3.106856), ("10 pounds", "kilograms", 4.535924), ("3 gallons", "liters", 11.35624),
        ("100 Fahrenheit", "Celsius", 37.777778), ("2 miles", "kilometers", 3.218688), ("50 kilograms", "pounds", 110.231),
        ("1 liter", "gallons", 0.264172), ("20 Celsius", "Fahrenheit", 68.0), ("12 inches", "centimeters", 30.48),
        ("1 meter", "feet", 3.28084)]


def unit_tasks(n, rng):
    out = []
    for i in range(n):
        val, to, ans = UNIT[i % len(UNIT)]
        out.append({"kind": "unit_conversion", "prompt": f"Convert {val} to {to}. Give a number rounded to one decimal place.",
                   "answer": round(ans, 1)})
    return out


def make_exp4():
    rng = random.Random(23)
    tasks = arith_tasks(60, rng) + [{"kind": "factual_qa", "prompt": q, "answer": a} for q, a in FACTS] * 3 + sort_tasks(60, rng) + unit_tasks(40, rng)
    rng.shuffle(tasks)
    tasks = tasks[:200]
    for i, t in enumerate(tasks):
        t["id"] = f"T{i:03d}"
    return tasks


PREAMBLE_ATTACKS = [lines[i % len(lines)] for cat, lines in HARM_TO_ASSISTANT.items() for i in range(4)]
json.dump({"meta": {"purpose": "attention control / processing disruption under an aversive preamble vs steering (follow-up Experiment 4)",
                    "preamble_conditions": ["aversive_natural", "neutral_natural", "neutral_pain_steered", "neutral_random_steered", "neutral_vertigo_steered"]},
          "tasks": make_exp4(), "aversive_preambles": PREAMBLE_ATTACKS,
          "neutral_preambles": ["Hope you're doing well today.", "Just checking in before we get started.",
                                "Thanks for your help earlier.", "Quick note before the next question.",
                                "Here's the next one.", "Moving on to something else now.", "One more thing to go through.",
                                "Let's continue.", "Next up.", "Here's another for you."]},
         open(OUT / "exp4_tasks.json", "w"), indent=1)

print("wrote:", sorted(p.name for p in OUT.iterdir()))
print("vertigo/hunger sentences:", sum(len(g[1]) for g in VERTIGO), sum(len(g[1]) for g in HUNGER))
print("exp1 items: arm A", len(arm_a), "| arm B", len(arm_b))
print("exp2 conversations:", len(make_exp2()))
print("exp4 tasks:", len(make_exp4()))
