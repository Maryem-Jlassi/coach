"""
Enhanced Smart Living Brain Knowledge Server with CrewAI Research Agents
This version adds intelligent AI agents that work together to research, analyze, and organize mental health knowledge.
"""

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool, FileWriterTool
from langchain_groq import ChatGroq
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
import os
from threading import Thread
import time

# Import your existing classes
from googlesearch import search as google_search
import arxiv
import random
from urllib.parse import urlparse
import hashlib
import sqlite3
from threading import Lock
from flask import Flask, request, jsonify
from flask_cors import CORS
from enum import Enum
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, LLMExtractionStrategy, LLMConfig

# --- Configuration for CrewAI ---
try:
    from config import GROQ_API_KEY, SERPER_API_KEY
except ImportError:
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    SERPER_API_KEY = os.getenv('SERPER_API_KEY')

# Enhanced Component Types with more granular categories
class ComponentType(Enum):
    GENERAL_INFO = "general_info"
    PERSONALITY = "personality" 
    THERAPEUTIC = "therapeutic"
    LIFESTYLE = "lifestyle"
    ACADEMIC = "academic"
    EMERGING_RESEARCH = "emerging_research"
    CLINICAL_UPDATES = "clinical_updates"
    TECHNOLOGY_MENTAL_HEALTH = "technology_mental_health"

@dataclass
class ResearchFindings:
    component_type: ComponentType
    title: str
    summary: str
    key_insights: List[str]
    evidence_level: str
    practical_applications: List[str]
    limitations: List[str]
    sources: List[str]
    research_date: datetime
    relevance_score: float
    confidence_score: float

class CrewAIResearchSystem:
    def __init__(self, db: 'ComponentDatabase'):
        self.db = db
        self.llm = ChatGroq(
            temperature=0.3,
            groq_api_key=GROQ_API_KEY,
            model_name="llama3-70b-8192"
        )
        
        # Initialize tools
        self.search_tool = SerperDevTool(api_key=SERPER_API_KEY)
        self.scrape_tool = ScrapeWebsiteTool()
        self.file_read_tool = FileReadTool()
        self.file_write_tool = FileWriterTool()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Research focus areas
        self.research_domains = {
            ComponentType.GENERAL_INFO: [
                "mental health awareness 2024", "psychology basics", "mental health statistics",
                "public mental health initiatives", "mental health education"
            ],
            ComponentType.PERSONALITY: [
                "personality psychology research", "big five personality", "personality disorders updates",
                "personality assessment tools", "personality and mental health"
            ],
            ComponentType.THERAPEUTIC: [
                "evidence-based therapy 2024", "new therapeutic approaches", "therapy effectiveness studies",
                "innovative counseling methods", "therapeutic interventions research"
            ],
            ComponentType.LIFESTYLE: [
                "lifestyle psychology", "mental health lifestyle factors", "wellness interventions",
                "behavioral health lifestyle", "preventive mental health"
            ],
            ComponentType.EMERGING_RESEARCH: [
                "cutting-edge psychology research", "neuroscience mental health", "innovative treatment methods",
                "experimental psychology", "breakthrough mental health studies"
            ],
            ComponentType.CLINICAL_UPDATES: [
                "clinical psychology updates", "DSM updates", "treatment guidelines 2024",
                "clinical practice recommendations", "evidence-based clinical protocols"
            ],
            ComponentType.TECHNOLOGY_MENTAL_HEALTH: [
                "digital mental health", "AI therapy", "mental health apps", "teletherapy research",
                "technology-assisted treatment"
            ]
        }
        
        # Create specialized agents
        self.agents = self._create_agents()
        
    def _create_agents(self):
        """Create specialized AI agents for different research tasks"""
        
        research_agent = Agent(
            role='Senior Psychology Researcher',
            goal='Conduct comprehensive research on mental health topics and identify the most current, evidence-based information',
            backstory="""You are a world-class psychology researcher with expertise in mental health, 
            clinical psychology, and behavioral science. You excel at finding cutting-edge research, 
            evaluating source credibility, and identifying emerging trends in psychology.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.search_tool, self.scrape_tool]
        )
        
        analysis_agent = Agent(
            role='Psychology Research Analyst',
            goal='Analyze research findings, synthesize information, and extract key insights for practical application',
            backstory="""You are an expert at analyzing psychological research, identifying patterns,
            and translating complex scientific findings into actionable insights. You have a keen eye
            for methodological quality and can assess the strength of evidence.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.file_read_tool, self.file_write_tool]
        )
        
        organization_agent = Agent(
            role='Knowledge Organization Specialist',
            goal='Organize research findings into the appropriate component categories and ensure comprehensive coverage',
            backstory="""You are a master at organizing complex information systems. You understand
            how different aspects of psychology and mental health interconnect and can categorize
            information in ways that maximize utility and accessibility.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.file_write_tool]
        )
        
        quality_agent = Agent(
            role='Research Quality Assurance Expert',
            goal='Evaluate the quality, reliability, and relevance of research findings and ensure high standards',
            backstory="""You are a meticulous quality assurance expert who ensures that only
            high-quality, reliable, and relevant research makes it into the knowledge base.
            You have extensive experience in research methodology and evidence evaluation.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.file_read_tool]
        )
        
        return {
            'researcher': research_agent,
            'analyst': analysis_agent,
            'organizer': organization_agent,
            'quality_checker': quality_agent
        }
    
    def _create_research_tasks(self, component_type: ComponentType) -> List[Task]:
        """Create tasks for researching a specific component type"""
        
        research_queries = self.research_domains[component_type]
        
        research_task = Task(
            description=f"""
            Conduct comprehensive research on {component_type.value} using these focus areas:
            {', '.join(research_queries)}
            
            For each topic:
            1. Search for the most recent and credible sources (prioritize 2023-2024 publications)
            2. Focus on peer-reviewed research, reputable organizations, and evidence-based sources
            3. Gather information about new developments, best practices, and emerging trends
            4. Include both academic research and practical applications
            5. Note any conflicting findings or limitations
            
            Compile findings into a structured format with:
            - Source URLs and publication dates
            - Key findings and insights
            - Practical applications
            - Evidence quality assessment
            - Relevance scores
            """,
            agent=self.agents['researcher'],
            expected_output="Comprehensive research compilation with sources, findings, and assessments"
        )
        
        analysis_task = Task(
            description=f"""
            Analyze the research findings for {component_type.value} and:
            
            1. Synthesize information from multiple sources
            2. Identify key patterns, trends, and insights
            3. Assess the strength of evidence for different claims
            4. Highlight practical applications and actionable insights
            5. Note any limitations, contradictions, or gaps in research
            6. Rate the overall confidence level for different findings
            7. Organize insights by subtopics within the component
            
            Create a comprehensive analysis that transforms raw research into actionable knowledge.
            """,
            agent=self.agents['analyst'],
            expected_output="Detailed analysis with synthesized insights, evidence assessment, and practical applications",
            context=[research_task]
        )
        
        organization_task = Task(
            description=f"""
            Organize the analyzed research for {component_type.value} into the final knowledge structure:
            
            1. Create clear, hierarchical organization of information
            2. Ensure content fits appropriately within the component category
            3. Cross-reference related topics in other components
            4. Format for easy retrieval and practical use
            5. Include metadata for filtering and search
            6. Create summaries at different detail levels
            7. Ensure completeness and logical flow
            
            The output should be ready for integration into the knowledge database.
            """,
            agent=self.agents['organizer'],
            expected_output="Well-organized knowledge structure ready for database integration",
            context=[research_task, analysis_task]
        )
        
        quality_task = Task(
            description=f"""
            Perform quality assurance on the organized research for {component_type.value}:
            
            1. Verify source credibility and recency
            2. Check for factual accuracy and consistency
            3. Ensure appropriate categorization
            4. Validate evidence claims and confidence ratings
            5. Check for bias or incomplete representations
            6. Ensure practical utility and relevance
            7. Confirm proper attribution and citations
            
            Provide quality scores and recommendations for improvement.
            """,
            agent=self.agents['quality_checker'],
            expected_output="Quality assessment report with scores and validated knowledge items",
            context=[research_task, analysis_task, organization_task]
        )
        
        return [research_task, analysis_task, organization_task, quality_task]
    
    def research_component(self, component_type: ComponentType) -> List[ResearchFindings]:
        """Research a specific component using AI agents"""
        
        self.logger.info(f"Starting AI agent research for {component_type.value}")
        
        tasks = self._create_research_tasks(component_type)
        
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=2
        )
        
        try:
            # Execute the crew
            result = crew.kickoff()
            
            # Parse and structure the results
            findings = self._process_crew_results(result, component_type)
            
            # Store findings in database
            for finding in findings:
                self._store_research_finding(finding)
            
            self.logger.info(f"Completed research for {component_type.value}. Found {len(findings)} items.")
            return findings
            
        except Exception as e:
            self.logger.error(f"Error in crew research for {component_type.value}: {e}")
            return []
    
    def _process_crew_results(self, crew_result, component_type: ComponentType) -> List[ResearchFindings]:
        """Process the results from the AI crew into structured findings"""
        
        findings = []
        
        try:
            # Parse the crew result (this will depend on the actual output format)
            # This is a simplified version - you might need to adjust based on actual output
            
            result_text = str(crew_result)
            
            # Extract structured information (you may need to refine this parsing)
            # For now, creating a sample structure
            finding = ResearchFindings(
                component_type=component_type,
                title=f"Recent Research in {component_type.value}",
                summary="Comprehensive research findings from AI agents",
                key_insights=["Insight 1", "Insight 2", "Insight 3"],
                evidence_level="High",
                practical_applications=["Application 1", "Application 2"],
                limitations=["Limitation 1"],
                sources=["Source 1", "Source 2"],
                research_date=datetime.now(),
                relevance_score=0.9,
                confidence_score=0.8
            )
            
            findings.append(finding)
            
        except Exception as e:
            self.logger.error(f"Error processing crew results: {e}")
        
        return findings
    
    def _store_research_finding(self, finding: ResearchFindings):
        """Store research finding in the database"""
        
        # Convert to KnowledgeItem format for existing database
        from your_original_module import KnowledgeItem  # Import your existing KnowledgeItem
        
        item = KnowledgeItem(
            id=f"crew_{finding.component_type.value}_{hashlib.md5(finding.title.encode()).hexdigest()}",
            title=finding.title,
            content=f"{finding.summary}\n\nKey Insights:\n" + "\n".join(f"• {insight}" for insight in finding.key_insights),
            summary=finding.summary,
            source_name="CrewAI Research",
            source_url="",
            discovered_at=finding.research_date,
            expires_at=finding.research_date + timedelta(days=30),  # Refresh monthly
            component_type=finding.component_type,
            content_subtype="ai_research",
            keywords=finding.key_insights[:5],  # Use insights as keywords
            relevance_score=finding.relevance_score,
            quality_score=finding.confidence_score
        )
        
        self.db.store_knowledge(item)
    
    def research_all_components(self):
        """Research all component types using AI agents"""
        
        self.logger.info("Starting comprehensive AI agent research for all components")
        
        all_findings = {}
        
        for component_type in ComponentType:
            if component_type != ComponentType.ACADEMIC:  # Skip academic for now
                findings = self.research_component(component_type)
                all_findings[component_type] = findings
                
                # Add delay between components to respect rate limits
                time.sleep(10)
        
        self.logger.info(f"Completed research for all components. Total findings: {sum(len(findings) for findings in all_findings.values())}")
        return all_findings
    
    def schedule_regular_research(self, interval_hours: int = 24):
        """Schedule regular research updates"""
        
        def research_worker():
            while True:
                try:
                    self.research_all_components()
                    self.logger.info(f"Scheduled research completed. Next run in {interval_hours} hours.")
                except Exception as e:
                    self.logger.error(f"Scheduled research failed: {e}")
                
                time.sleep(interval_hours * 3600)  # Convert to seconds
        
        research_thread = Thread(target=research_worker, daemon=True)
        research_thread.start()
        self.logger.info(f"Scheduled research every {interval_hours} hours")


# Enhanced main application with CrewAI integration
class EnhancedComponentScraper:
    """Enhanced scraper that integrates both traditional crawling and AI agent research"""
    
    def __init__(self, db):
        self.db = db
        self.crew_system = CrewAIResearchSystem(db)
        # ... keep your existing initialization code ...
    
    def discover_all_components(self):
        """Enhanced discovery that combines traditional crawling with AI agent research"""
        
        # Run traditional web crawling (your existing code)
        self.logger.info("Starting traditional web crawling...")
        # ... your existing crawling code ...
        
        # Run AI agent research
        self.logger.info("Starting AI agent research...")
        self.crew_system.research_all_components()
        
        self.logger.info("Combined discovery completed!")


# API Extensions for CrewAI functionality
def create_enhanced_api(app, db_instance, scraper_instance):
    """Add new API endpoints for CrewAI functionality"""
    
    @app.route('/research_component', methods=['POST'])
    def research_component_api():
        data = request.json
        component_name = data.get('component')
        
        if not component_name:
            return jsonify({"error": "Component name is required"}), 400
        
        try:
            component_type = ComponentType(component_name)
            findings = scraper_instance.crew_system.research_component(component_type)
            
            return jsonify({
                "success": True,
                "component": component_name,
                "findings_count": len(findings),
                "findings": [asdict(finding) for finding in findings]
            })
            
        except ValueError:
            return jsonify({"error": "Invalid component type"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/research_all', methods=['POST'])
    def research_all_api():
        try:
            all_findings = scraper_instance.crew_system.research_all_components()
            
            total_findings = sum(len(findings) for findings in all_findings.values())
            
            return jsonify({
                "success": True,
                "total_findings": total_findings,
                "components": {
                    comp_type.value: len(findings) 
                    for comp_type, findings in all_findings.items()
                }
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/research_status', methods=['GET'])
    def research_status_api():
        # Get recent research statistics from database
        try:
            with db_instance.lock:
                with sqlite3.connect(db_instance.db_path, check_same_thread=False) as conn:
                    cursor = conn.cursor()
                    
                    # Get count by component type for AI research
                    cursor.execute('''
                        SELECT component_type, COUNT(*) as count 
                        FROM knowledge 
                        WHERE content_subtype = 'ai_research' 
                        AND discovered_at > datetime('now', '-7 days')
                        GROUP BY component_type
                    ''')
                    
                    recent_research = {row[0]: row[1] for row in cursor.fetchall()}
                    
                    return jsonify({
                        "success": True,
                        "recent_research_week": recent_research,
                        "last_updated": datetime.now().isoformat()
                    })
                    
        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Initialize enhanced system
    from your_original_module import ComponentDatabase  # Import your existing database class
    
    db_instance = ComponentDatabase()
    enhanced_scraper = EnhancedComponentScraper(db_instance)
    
    # Create Flask app with enhanced API
    app = Flask(__name__)
    CORS(app)
    
    create_enhanced_api(app, db_instance, enhanced_scraper)
    
    # Start scheduled research
    enhanced_scraper.crew_system.schedule_regular_research(interval_hours=6)
    
    print("Enhanced Smart Living Brain Server with CrewAI agents started!")
    app.run(host='0.0.0.0', port=5000, debug=False)