from agents.crl import CRLAgent
from agents.gciql import GCIQLAgent
from agents.hiql import HIQLAgent
from agents.sac import SACAgent

agents = dict(
    crl=CRLAgent,
    gciql=GCIQLAgent,
    hiql=HIQLAgent,
    sac=SACAgent,
)
