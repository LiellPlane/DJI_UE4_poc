from dataclasses import dataclass

@dataclass
class base_out_msg:
    my_id: str
    timestamp: str


@dataclass
class hit_report_light:
    my_id: str
    hit_id: str
    target_id: str


@dataclass
class hit_report_heavy(hit_report_light):
    capture_image: str
    insult_text: str
    insult_snd: str

 
@dataclass
class error:
    error_str: str


@dataclass
class heartbeat(base_out_msg):
    heartbeat: str

