from memory_client import write_memory

write_memory(
    entity='INC0000077', fact_type='priority', value='2-High',
    agent_id='delivery_agent', extraction_type='inferred',
    confidence=0.90, source_file='test_close_call'
)

write_memory(
    entity='INC0000077', fact_type='priority', value='4-Low',
    agent_id='billing_agent', extraction_type='inferred',
    confidence=0.85, source_file='test_close_call'
)

print("Done writing test conflict facts.")