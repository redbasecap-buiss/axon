use std::io::{self, stdout};
use std::path::Path;

use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    prelude::CrosstermBackend,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap},
    Frame, Terminal,
};

use crate::db::{Brain, Entity, Fact};

#[derive(Debug, PartialEq, Eq)]
enum Mode {
    Normal,
    Search,
    FeedUrl,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Panel {
    Entities,
    Relations,
}

struct App {
    brain: Brain,
    entities: Vec<Entity>,
    filtered: Vec<usize>,
    list_state: ListState,
    rel_state: ListState,
    mode: Mode,
    panel: Panel,
    search_query: String,
    feed_url: String,
    status_msg: String,
    show_stats: bool,
    cached_facts: Vec<Fact>,
    cached_relations: Vec<(String, String, String, f64)>,
}

impl App {
    fn new(brain: Brain) -> Self {
        let entities = brain.all_entities().unwrap_or_default();
        let filtered: Vec<usize> = (0..entities.len()).collect();
        let mut list_state = ListState::default();
        if !filtered.is_empty() {
            list_state.select(Some(0));
        }
        let mut app = App {
            brain,
            entities,
            filtered,
            list_state,
            rel_state: ListState::default(),
            mode: Mode::Normal,
            panel: Panel::Entities,
            search_query: String::new(),
            feed_url: String::new(),
            status_msg: String::new(),
            show_stats: false,
            cached_facts: Vec::new(),
            cached_relations: Vec::new(),
        };
        app.update_cache();
        app
    }

    fn reload_entities(&mut self) {
        self.entities = self.brain.all_entities().unwrap_or_default();
        self.apply_filter();
    }

    fn apply_filter(&mut self) {
        if self.search_query.is_empty() {
            self.filtered = (0..self.entities.len()).collect();
        } else {
            let q = self.search_query.to_lowercase();
            self.filtered = self
                .entities
                .iter()
                .enumerate()
                .filter(|(_, e)| e.name.to_lowercase().contains(&q))
                .map(|(i, _)| i)
                .collect();
        }
        if self.filtered.is_empty() {
            self.list_state.select(None);
        } else {
            self.list_state.select(Some(0));
        }
        self.update_cache();
    }

    fn selected_entity(&self) -> Option<&Entity> {
        self.list_state
            .selected()
            .and_then(|i| self.filtered.get(i))
            .and_then(|&idx| self.entities.get(idx))
    }

    fn update_cache(&mut self) {
        if let Some(entity) = self.selected_entity() {
            let eid = entity.id;
            self.cached_facts = self.brain.get_facts_for(eid).unwrap_or_default();
            self.cached_relations = self.brain.get_relations_for(eid).unwrap_or_default();
            if !self.cached_relations.is_empty() && self.rel_state.selected().is_none() {
                self.rel_state.select(Some(0));
            }
        } else {
            self.cached_facts.clear();
            self.cached_relations.clear();
            self.rel_state.select(None);
        }
    }

    fn move_selection(&mut self, delta: isize) {
        match self.panel {
            Panel::Entities => {
                if self.filtered.is_empty() {
                    return;
                }
                let cur = self.list_state.selected().unwrap_or(0) as isize;
                let next = (cur + delta).clamp(0, self.filtered.len() as isize - 1) as usize;
                self.list_state.select(Some(next));
                self.rel_state.select(if self.cached_relations.is_empty() {
                    None
                } else {
                    Some(0)
                });
                self.update_cache();
            }
            Panel::Relations => {
                if self.cached_relations.is_empty() {
                    return;
                }
                let cur = self.rel_state.selected().unwrap_or(0) as isize;
                let next =
                    (cur + delta).clamp(0, self.cached_relations.len() as isize - 1) as usize;
                self.rel_state.select(Some(next));
            }
        }
    }

    fn jump_top(&mut self) {
        match self.panel {
            Panel::Entities => {
                if !self.filtered.is_empty() {
                    self.list_state.select(Some(0));
                    self.update_cache();
                }
            }
            Panel::Relations => {
                if !self.cached_relations.is_empty() {
                    self.rel_state.select(Some(0));
                }
            }
        }
    }

    fn jump_bottom(&mut self) {
        match self.panel {
            Panel::Entities => {
                if !self.filtered.is_empty() {
                    self.list_state.select(Some(self.filtered.len() - 1));
                    self.update_cache();
                }
            }
            Panel::Relations => {
                if !self.cached_relations.is_empty() {
                    self.rel_state.select(Some(self.cached_relations.len() - 1));
                }
            }
        }
    }

    fn drill_into(&mut self) {
        if self.panel == Panel::Relations {
            if let Some(sel) = self.rel_state.selected() {
                if let Some((subj, _pred, obj, _conf)) = self.cached_relations.get(sel) {
                    let current_name = self
                        .selected_entity()
                        .map(|e| e.name.clone())
                        .unwrap_or_default();
                    let target = if *subj == current_name {
                        obj.clone()
                    } else {
                        subj.clone()
                    };
                    if let Some(pos) = self
                        .filtered
                        .iter()
                        .position(|&idx| self.entities.get(idx).is_some_and(|e| e.name == target))
                    {
                        self.list_state.select(Some(pos));
                        self.panel = Panel::Entities;
                        self.update_cache();
                        return;
                    }
                    self.search_query.clear();
                    self.apply_filter();
                    if let Some(pos) = self
                        .filtered
                        .iter()
                        .position(|&idx| self.entities.get(idx).is_some_and(|e| e.name == target))
                    {
                        self.list_state.select(Some(pos));
                        self.panel = Panel::Entities;
                        self.update_cache();
                    }
                }
            }
        }
    }
}

fn confidence_color(c: f64) -> Color {
    if c >= 0.7 {
        Color::Green
    } else if c >= 0.4 {
        Color::Yellow
    } else {
        Color::Red
    }
}

fn confidence_span(c: f64) -> Span<'static> {
    Span::styled(
        format!("{:.0}%", c * 100.0),
        Style::default().fg(confidence_color(c)),
    )
}

fn draw(f: &mut Frame, app: &mut App) {
    let size = f.area();
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(3), Constraint::Length(1)])
        .split(size);
    let panels = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(45),
            Constraint::Percentage(30),
        ])
        .split(main_chunks[0]);
    draw_entity_list(f, app, panels[0]);
    draw_details(f, app, panels[1]);
    draw_relations(f, app, panels[2]);
    draw_status_bar(f, app, main_chunks[1]);
    if app.show_stats {
        draw_stats_overlay(f, app, size);
    }
    if app.mode == Mode::Search || app.mode == Mode::FeedUrl {
        draw_input_bar(f, app, size);
    }
}

fn draw_entity_list(f: &mut Frame, app: &mut App, area: Rect) {
    let is_focused = app.panel == Panel::Entities;
    let title = if app.search_query.is_empty() {
        format!(" Entities ({}) ", app.filtered.len())
    } else {
        format!(
            " Entities [/{}] ({}) ",
            app.search_query,
            app.filtered.len()
        )
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(if is_focused {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        });
    let items: Vec<ListItem> = app
        .filtered
        .iter()
        .filter_map(|&idx| app.entities.get(idx))
        .map(|e| {
            ListItem::new(Line::from(vec![
                Span::raw(e.name.clone()),
                Span::raw(" "),
                confidence_span(e.confidence),
            ]))
        })
        .collect();
    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");
    f.render_stateful_widget(list, area, &mut app.list_state);
}

fn draw_details(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().title(" Details ").borders(Borders::ALL);
    let entity = match app.selected_entity() {
        Some(e) => e,
        None => {
            f.render_widget(Paragraph::new("No entity selected").block(block), area);
            return;
        }
    };
    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled(
                entity.name.clone(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("[{}]", entity.entity_type),
                Style::default().fg(Color::Magenta),
            ),
        ]),
        Line::from(vec![
            Span::raw("Confidence: "),
            confidence_span(entity.confidence),
            Span::raw(format!("  Seen: {}x", entity.access_count)),
        ]),
        Line::from(format!(
            "First: {}  Last: {}",
            entity.first_seen.format("%Y-%m-%d"),
            entity.last_seen.format("%Y-%m-%d"),
        )),
        Line::raw(""),
    ];
    if !app.cached_facts.is_empty() {
        lines.push(Line::styled(
            "Facts:",
            Style::default().add_modifier(Modifier::UNDERLINED),
        ));
        for fact in &app.cached_facts {
            lines.push(Line::from(vec![
                Span::styled(format!("  {} ", fact.key), Style::default().fg(Color::Cyan)),
                Span::raw("= "),
                Span::raw(fact.value.clone()),
                Span::raw(" "),
                confidence_span(fact.confidence),
            ]));
        }
    }
    f.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        area,
    );
}

fn draw_relations(f: &mut Frame, app: &mut App, area: Rect) {
    let is_focused = app.panel == Panel::Relations;
    let block = Block::default()
        .title(format!(" Relations ({}) ", app.cached_relations.len()))
        .borders(Borders::ALL)
        .border_style(if is_focused {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::DarkGray)
        });
    let current_name = app
        .selected_entity()
        .map(|e| e.name.clone())
        .unwrap_or_default();
    let items: Vec<ListItem> = app
        .cached_relations
        .iter()
        .map(|(subj, pred, obj, conf)| {
            let other = if *subj == current_name {
                obj.clone()
            } else {
                subj.clone()
            };
            ListItem::new(Line::from(vec![
                Span::styled(pred.clone(), Style::default().fg(Color::Yellow)),
                Span::raw(" -> "),
                Span::raw(other),
                Span::raw(" "),
                confidence_span(*conf),
            ]))
        })
        .collect();
    let list = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");
    f.render_stateful_widget(list, area, &mut app.rel_state);
}

fn draw_status_bar(f: &mut Frame, app: &App, area: Rect) {
    let stats = app.brain.stats().unwrap_or(crate::db::Stats {
        entity_count: 0,
        relation_count: 0,
        fact_count: 0,
        source_count: 0,
        db_size: "?".into(),
    });
    let msg = if app.status_msg.is_empty() {
        String::new()
    } else {
        format!("  |  {}", app.status_msg)
    };
    let bar = Paragraph::new(Line::from(vec![
        Span::styled(" axon ", Style::default().fg(Color::Black).bg(Color::Cyan)),
        Span::raw(format!(
            "  {} entities  {} relations  {} facts  {}{}",
            stats.entity_count, stats.relation_count, stats.fact_count, stats.db_size, msg
        )),
    ]))
    .style(Style::default().bg(Color::DarkGray));
    f.render_widget(bar, area);
}

fn draw_stats_overlay(f: &mut Frame, app: &App, area: Rect) {
    let stats = app.brain.stats().unwrap_or(crate::db::Stats {
        entity_count: 0,
        relation_count: 0,
        fact_count: 0,
        source_count: 0,
        db_size: "?".into(),
    });
    let w = 40u16.min(area.width.saturating_sub(4));
    let h = 10u16.min(area.height.saturating_sub(4));
    let x = (area.width.saturating_sub(w)) / 2;
    let y = (area.height.saturating_sub(h)) / 2;
    let overlay = Rect::new(x, y, w, h);
    let text = vec![
        Line::styled(
            "Brain Statistics",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Line::raw(""),
        Line::raw(format!("  Entities:   {}", stats.entity_count)),
        Line::raw(format!("  Relations:  {}", stats.relation_count)),
        Line::raw(format!("  Facts:      {}", stats.fact_count)),
        Line::raw(format!("  Sources:    {}", stats.source_count)),
        Line::raw(format!("  DB size:    {}", stats.db_size)),
        Line::raw(""),
        Line::styled("  Press 's' to close", Style::default().fg(Color::DarkGray)),
    ];
    let block = Block::default()
        .title(" Stats ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    f.render_widget(Clear, overlay);
    f.render_widget(Paragraph::new(text).block(block), overlay);
}

fn draw_input_bar(f: &mut Frame, app: &App, area: Rect) {
    let (label, content) = match app.mode {
        Mode::Search => ("/", &app.search_query),
        Mode::FeedUrl => ("Feed URL: ", &app.feed_url),
        Mode::Normal => return,
    };
    let h = 3u16;
    let y = area.height.saturating_sub(h + 1);
    let input_area = Rect::new(0, y, area.width, h);
    let block = Block::default().borders(Borders::ALL).title(" Input ");
    let text = format!("{label}{content}_");
    f.render_widget(Clear, input_area);
    f.render_widget(Paragraph::new(text).block(block), input_area);
}

pub fn run(db_path: &Path) -> anyhow::Result<()> {
    let brain = Brain::open(db_path)?;
    let mut app = App::new(brain);
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    let result = run_loop(&mut terminal, &mut app);
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    result
}

fn run_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> anyhow::Result<()> {
    loop {
        terminal.draw(|f| draw(f, app))?;
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
                return Ok(());
            }
            match app.mode {
                Mode::Search => match key.code {
                    KeyCode::Esc => {
                        app.mode = Mode::Normal;
                        app.search_query.clear();
                        app.apply_filter();
                    }
                    KeyCode::Enter => {
                        app.mode = Mode::Normal;
                    }
                    KeyCode::Backspace => {
                        app.search_query.pop();
                        app.apply_filter();
                    }
                    KeyCode::Char(c) => {
                        app.search_query.push(c);
                        app.apply_filter();
                    }
                    _ => {}
                },
                Mode::FeedUrl => match key.code {
                    KeyCode::Esc => {
                        app.mode = Mode::Normal;
                        app.feed_url.clear();
                    }
                    KeyCode::Enter => {
                        let url = app.feed_url.clone();
                        app.feed_url.clear();
                        app.mode = Mode::Normal;
                        if !url.is_empty() {
                            app.status_msg = format!("Feeding {url}...");
                            terminal.draw(|f| draw(f, app))?;
                            match feed_url_sync(&app.brain, &url) {
                                Ok(msg) => {
                                    app.status_msg = msg;
                                    app.reload_entities();
                                }
                                Err(e) => app.status_msg = format!("Error: {e}"),
                            }
                        }
                    }
                    KeyCode::Backspace => {
                        app.feed_url.pop();
                    }
                    KeyCode::Char(c) => {
                        app.feed_url.push(c);
                    }
                    _ => {}
                },
                Mode::Normal => match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('j') | KeyCode::Down => app.move_selection(1),
                    KeyCode::Char('k') | KeyCode::Up => app.move_selection(-1),
                    KeyCode::Char('g') => app.jump_top(),
                    KeyCode::Char('G') => app.jump_bottom(),
                    KeyCode::Char('/') => {
                        app.mode = Mode::Search;
                        app.search_query.clear();
                    }
                    KeyCode::Char('f') => {
                        app.mode = Mode::FeedUrl;
                        app.feed_url.clear();
                    }
                    KeyCode::Char('s') => {
                        app.show_stats = !app.show_stats;
                    }
                    KeyCode::Tab => {
                        app.panel = match app.panel {
                            Panel::Entities => Panel::Relations,
                            Panel::Relations => Panel::Entities,
                        };
                    }
                    KeyCode::Enter => app.drill_into(),
                    _ => {}
                },
            }
        }
    }
}

fn feed_url_sync(brain: &Brain, url: &str) -> anyhow::Result<String> {
    let rt = tokio::runtime::Runtime::new()?;
    let text = rt.block_on(crate::crawler::fetch_and_extract(url))?;
    let extracted = crate::nlp::process_text(&text, url);
    let (entities, relations, facts) = brain.learn(&extracted)?;
    Ok(format!(
        "Fed! +{entities} entities, +{relations} relations, +{facts} facts"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_color_green() {
        assert_eq!(confidence_color(0.8), Color::Green);
    }

    #[test]
    fn test_confidence_color_yellow() {
        assert_eq!(confidence_color(0.5), Color::Yellow);
    }

    #[test]
    fn test_confidence_color_red() {
        assert_eq!(confidence_color(0.2), Color::Red);
    }

    #[test]
    fn test_app_new_empty_brain() {
        let brain = Brain::open_in_memory().unwrap();
        let app = App::new(brain);
        assert!(app.entities.is_empty());
        assert!(app.filtered.is_empty());
        assert_eq!(app.mode, Mode::Normal);
    }

    #[test]
    fn test_app_with_entities() {
        let brain = Brain::open_in_memory().unwrap();
        brain.upsert_entity("Rust", "language").unwrap();
        brain.upsert_entity("Python", "language").unwrap();
        let app = App::new(brain);
        assert_eq!(app.entities.len(), 2);
        assert_eq!(app.filtered.len(), 2);
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn test_app_filter() {
        let brain = Brain::open_in_memory().unwrap();
        brain.upsert_entity("Rust", "language").unwrap();
        brain.upsert_entity("Python", "language").unwrap();
        let mut app = App::new(brain);
        app.search_query = "Rust".to_string();
        app.apply_filter();
        assert_eq!(app.filtered.len(), 1);
    }

    #[test]
    fn test_app_move_selection() {
        let brain = Brain::open_in_memory().unwrap();
        brain.upsert_entity("A", "t").unwrap();
        brain.upsert_entity("B", "t").unwrap();
        brain.upsert_entity("C", "t").unwrap();
        let mut app = App::new(brain);
        assert_eq!(app.list_state.selected(), Some(0));
        app.move_selection(1);
        assert_eq!(app.list_state.selected(), Some(1));
        app.move_selection(-1);
        assert_eq!(app.list_state.selected(), Some(0));
        app.move_selection(-1);
        assert_eq!(app.list_state.selected(), Some(0));
    }

    #[test]
    fn test_app_jump_top_bottom() {
        let brain = Brain::open_in_memory().unwrap();
        brain.upsert_entity("A", "t").unwrap();
        brain.upsert_entity("B", "t").unwrap();
        brain.upsert_entity("C", "t").unwrap();
        let mut app = App::new(brain);
        app.jump_bottom();
        assert_eq!(app.list_state.selected(), Some(2));
        app.jump_top();
        assert_eq!(app.list_state.selected(), Some(0));
    }
}
