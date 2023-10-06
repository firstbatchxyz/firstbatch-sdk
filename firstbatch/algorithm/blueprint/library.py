"""Factory Algorithms for Blueprint"""
# Navigable UX

# Transform your user experience into a navigable journey by leveraging user interactions.
# Every user action contributes to shaping their unique experience.

# Target : Anonymous sessions to navigable experiences

# Algo logic description : Trying to serve various content until receiving first signal from users.

# Then crafting the experience sharply regarding signals. Gradually providing more space for exploration if user
# keeps loosing interest in current topics.

# Potential KPIs : Any engagement metric, Time spent on application or Bounce rate,
# Conversion rate of anonymous sessions.

Navigable_UX = '''{
    "nodes": [
        {"name": "Exploration", "batch_type": "random", "params": {"last_n":8}},
        {"name": "Browsing", "batch_type": "sampled", "params": {"n_topics":12,"last_n":8 }},
        {"name": "Discovery", "batch_type": "personalized", "params": {"r" : 0.2, "mu" : 0.5, "alpha" : 0.7, "apply_threshold": 0.3, "apply_mmr" :true}},
        {"name": "Dedicated", "batch_type": "personalized", "params": {"r" : 0.1, "mu" : 0.2, "alpha" : 0.4, "apply_threshold": 0.5,"apply_mmr" :true, "last_n":5}},
        {"name": "Focus", "batch_type": "personalized", "params": {"r" : 0.1, "mu" : 0.05, "alpha" : 0.1, "apply_threshold": 0.6, "apply_mmr" :false, "last_n":4}},
        {"name": "Hyper_Focus", "batch_type": "personalized", "params": {"r" : 0, "mu" : 0, "alpha" : 0, "apply_threshold": 0.7, "apply_mmr" :false, "last_n":2}}
    ],
    "edges": [
        {"name": "edge1", "edge_type": "DEFAULT", "start": "Exploration", "end": "Hyper_Focus"},
        {"name": "edge2", "edge_type": "DEFAULT", "start": "Browsing", "end": "Hyper_Focus"},
        {"name": "edge3", "edge_type": "DEFAULT", "start": "Discovery", "end": "Hyper_Focus"},
        {"name": "edge4", "edge_type": "DEFAULT", "start": "Dedicated", "end": "Hyper_Focus"},
        {"name": "edge5", "edge_type": "DEFAULT", "start": "Focus", "end": "Hyper_Focus"},
        {"name": "edge6", "edge_type": "DEFAULT", "start": "Hyper_Focus", "end": "Hyper_Focus"},
        {"name": "edge7", "edge_type": "BATCH", "start": "Exploration", "end": "Browsing"},
        {"name": "edge8", "edge_type": "BATCH", "start": "Browsing", "end": "Browsing"},
        {"name": "edge9", "edge_type": "BATCH", "start": "Discovery", "end": "Discovery"},
        {"name": "edge10", "edge_type": "BATCH", "start": "Dedicated", "end": "Discovery"},
        {"name": "edge11", "edge_type": "BATCH", "start": "Focus", "end": "Dedicated"},
        {"name": "edge12", "edge_type": "BATCH", "start": "Hyper_Focus", "end": "Focus"}
    ]
}'''

# Individually Crafted Recommendations

# Offer users not only similar but also adjacent items in a personalized manner.

# This approach allows users to discover new and relevant content on their own terms,
# enhancing their exploration and satisfaction.

# Target : Increase up-sell and help you to improve average order value.

# Algo logic description : Making highly focused recommendations after first interaction.

# But enable users to explore more items from a wider perspective to keep users within
# recommendations space until they find something to add their cart.

# Potential KPIs : Up-sell and cross-sell metrics. Average Order Value. Number of items per order.


Individually_Crafted_Recommendations = '''{
    "nodes": [
        {"name": "Recommendation", "batch_type": "personalized", "params": {"r" : 0, "mu" : 0.0, "alpha" : 0, "apply_threshold": 0.7, "apply_mmr" :false, "last_n":1}},
        {"name": "Expansion", "batch_type": "personalized", "params": {"r" : 0, "mu" : 0.05, "alpha" : 0.4, "apply_threshold": 0.7, "apply_mmr" :true, "last_n":2}},
        {"name": "Discovery", "batch_type": "personalized", "params": {"r" : 0, "mu" : 0.1, "alpha" : 1, "apply_threshold": 0.6, "apply_mmr" :true, "last_n":4}}
    ],
    "edges": [
        {"name": "edge1", "edge_type": "DEFAULT", "start": "Discovery", "end": "Recommendation"},
        {"name": "edge2", "edge_type": "DEFAULT", "start": "Expansion", "end": "Recommendation"},
        {"name": "edge3", "edge_type": "DEFAULT", "start": "Recommendation", "end": "Recommendation"},
        {"name": "edge4", "edge_type": "BATCH", "start": "Recommendation", "end": "Expansion"},
        {"name": "edge5", "edge_type": "BATCH", "start": "Expansion", "end": "Discovery"},
        {"name": "edge6", "edge_type": "BATCH", "start": "Discovery", "end": "Discovery"}
    ]
}'''

# Unique Journeys

# Enable users to access the right content from the very beginning by
# tailoring their experience based on their starting point.

# Target : Shape user journey from the very beginning. Might be best for
# traffic source or seasonal campaigns based welcoming, and recurring visitor experiences.

# Algo logic description : Providing focused content starting from first
# load by utilizing user embeddings from previous sessions or adding seasonal effect to the experience.

# For example adding summer collection as bias during summer. Then letting navigate themselves
# just as we do in the Navigable_UX algorithm.

# Potential KPIs : Up-Any engagement metric, Time spent before first interaction,
# conversion rate of recurring visitors, Time spent on application or Bounce rate.

Unique_Journeys = '''{
        "nodes": [
            {"name": "Welcome", "batch_type": "biased", "params": {"r" : 0.2, "mu" : 0.2, "alpha" : 0.4, "apply_threshold": 0.7, "apply_mmr" :false, "last_n":5}},
            {"name": "Exploration", "batch_type": "personalized", "params": {"r" : 0.3, "mu" : 0.6, "alpha" : 0.7, "apply_threshold": 0.3, "apply_mmr" :true, "last_n":8}},
            {"name": "Discovery", "batch_type": "personalized", "params": {"r" : 0.2, "mu" : 0.4, "alpha" : 0.5, "apply_threshold": 0.3, "apply_mmr" :true, "last_n":6}},
            {"name": "Dedicated", "batch_type": "personalized", "params": {"r" : 0.1, "mu" : 0.2, "alpha" : 0.4, "apply_threshold": 0.5,"apply_mmr" :true, "last_n":5}},
            {"name": "Focus", "batch_type": "personalized", "params": {"r" : 0.1, "mu" : 0.05, "alpha" : 0.1, "apply_threshold": 0.6, "apply_mmr" :false, "last_n":4}},
            {"name": "Hyper_Focus", "batch_type": "personalized", "params": {"r" : 0, "mu" : 0, "alpha" : 0, "apply_threshold": 0.7, "apply_mmr" :false, "last_n":2}}
        ],
        "edges": [
            {"name": "edge1", "edge_type": "DEFAULT", "start": "Welcome", "end": "Hyper_Focus"},
            {"name": "edge2", "edge_type": "DEFAULT", "start": "Exploration", "end": "Hyper_Focus"},
            {"name": "edge3", "edge_type": "DEFAULT", "start": "Discovery", "end": "Hyper_Focus"},
            {"name": "edge4", "edge_type": "DEFAULT", "start": "Dedicated", "end": "Hyper_Focus"},
            {"name": "edge5", "edge_type": "DEFAULT", "start": "Focus", "end": "Hyper_Focus"},
            {"name": "edge6", "edge_type": "DEFAULT", "start": "Hyper_Focus", "end": "Hyper_Focus"},
            {"name": "edge7", "edge_type": "BATCH", "start": "Welcome", "end": "Exploration"},
            {"name": "edge8", "edge_type": "BATCH", "start": "Exploration", "end": "Welcome"},
            {"name": "edge9", "edge_type": "BATCH", "start": "Discovery", "end": "Exploration"},
            {"name": "edge10", "edge_type": "BATCH", "start": "Dedicated", "end": "Discovery"},
            {"name": "edge11", "edge_type": "BATCH", "start": "Focus", "end": "Dedicated"},
            {"name": "edge12", "edge_type": "BATCH", "start": "Hyper_Focus", "end": "Focus"}
        ]
    }'''


# Not User Targeting but User-Centric Promoted Content Curations

# Shift away from conventional targeting techniques and embrace a user-centric approach to deliver promoted
# items or ads in a captivating format.

# This approach allows users to actively influence the curation of promoted content,
# ensuring it aligns seamlessly with their preferences and resulting in a highly interactive and enjoyable experience.

# Target : Not force your users to see irrelevant promoted content but provide an engaging campaign discovery.

# People ignore ads because targeting only pollutes feeds. Therefore the aim is improveing campaign
# CTR by providing true content for right users at the right time.

# Algo logic description : Promoting contents in hyper-personalized manner by keeping curation
# focused after first interaction to forever.

# Potential KPIs : CTR

User_Centric_Promoted_Content_Curations = '''{
    "nodes": [
        {"name": "Exploration", "batch_type": "sampled", "params": {"n_topics":8,"last_n": 3}},
        {"name": "Curated", "batch_type": "personalized", "params": {"r" : 0, "mu" : 0.1, "alpha" : 0.3, "apply_threshold": 0.6, "apply_mmr" :false, "last_n":1}}
    ],
    "edges": [
        {"name": "edge1", "edge_type": "DEFAULT", "start": "Exploration", "end": "Curated"},
        {"name": "edge2", "edge_type": "DEFAULT", "start": "Curated", "end": "Curated"},
        {"name": "edge3", "edge_type": "BATCH", "start": "Exploration", "end": "Exploration"},
        {"name": "edge4", "edge_type": "BATCH", "start": "Curated", "end": "Curated"}
    ]
}'''

# User-Intent AI Agents
# Empower your AI agents with real-time insights into user intentions, derived from their interactions.

# This infusion of user intent brings intimacy to AI-driven experiences,
# making users feel more connected and understood.

# Target : Serving personal AI assistance that reflects user interactions that are not restricted with prompts.

# Algo logic description : Not giving space for false navigation and keeping
# the AI agent as much as closer to user intentions.

# Because people are being demotivated by hallucinated conversations with AI too fast.

# Potential KPIs : Time spent with AI agents, Chat Rating, Conversion rate through AI agents.

User_Intent_AI_Agents = '''{
    "nodes": [
        {"name": "Welcome", "batch_type": "biased", "params": {"r" : 0, "mu" : 0.1, "alpha" : 0.4, "apply_threshold": 0.8, "apply_mmr" :false, "last_n":12}},
        {"name": "Expansion", "batch_type": "personalized", "params": {"r" : 0, "mu" : 0.2, "alpha" : 0.6, "apply_threshold": 0.6, "apply_mmr" :true, "last_n":12}},
        {"name": "Exploration", "batch_type": "personalized", "params": {"r" : 0, "mu" : 0.5, "alpha" : 0.6, "apply_threshold": 0.5,"apply_mmr" :true, "last_n":12}},
        {"name": "Focus", "batch_type": "personalized", "params": {"r" : 0, "mu" : 0, "alpha" : 0, "apply_threshold": 0.8, "apply_mmr" :false, "last_n":6}}
    ],
    "edges": [
        {"name": "edge1", "edge_type": "DEFAULT", "start": "Welcome", "end": "Hyper_Focus"},
        {"name": "edge2", "edge_type": "DEFAULT", "start": "Expansion", "end": "Hyper_Focus"},
        {"name": "edge3", "edge_type": "DEFAULT", "start": "Exploration", "end": "Hyper_Focus"},
        {"name": "edge4", "edge_type": "DEFAULT", "start": "Focus", "end": "Hyper_Focus"},
        {"name": "edge1", "edge_type": "BATCH", "start": "Welcome", "end": "Welcome"},
        {"name": "edge2", "edge_type": "BATCH", "start": "Expansion", "end": "Exploration"},
        {"name": "edge3", "edge_type": "BATCH", "start": "Exploration", "end": "Exploration"},
        {"name": "edge4", "edge_type": "BATCH", "start": "Focus", "end": "Expansion"}
    ]
}'''

lookup = {
    "Unique_Journeys".upper(): Unique_Journeys,
    "User_Centric_Promoted_Content_Curations".upper(): User_Centric_Promoted_Content_Curations,
    "User_Intent_AI_Agents".upper(): User_Intent_AI_Agents,
    "Individually_Crafted_Recommendations".upper(): Individually_Crafted_Recommendations,
    "Navigable_UX".upper(): Navigable_UX
}